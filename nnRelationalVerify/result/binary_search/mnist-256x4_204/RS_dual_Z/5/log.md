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
execution time: IAR + LP analysis = 1.11 + 2.88 = 3.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.9162456, upper bound: 1.9162456


# Binary Search by BASE starts (time budget: 2696.01 seconds, max iter: 100)

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
Binary search time: 15.37 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2680.64 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9135725, upper bound: 1.9129797
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9129798, upper bound: 1.9135718
time: 2.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.70
Output dim: 6, lower bound: -1.9135725, upper bound: 1.9129797
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.70
Output dim: 6, lower bound: -1.9129798, upper bound: 1.9135718

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795927
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795927
time: 2.31 seconds

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8800166
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8800166
time: 2.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795927
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795927
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8800166
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8800166

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795587
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8799990, upper bound: 1.8795927
time: 2.06 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795587
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8799990, upper bound: 1.8795927
time: 2.06 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8799990
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795587, upper bound: 1.8800165
time: 2.05 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8799990
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8795587, upper bound: 1.8800166
time: 2.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795587
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8799990, upper bound: 1.8795927
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8800166, upper bound: 1.8795587
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8799990, upper bound: 1.8795927
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8799990
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8795587, upper bound: 1.8800165
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8795927, upper bound: 1.8799990
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.19
Output dim: 6, lower bound: -1.8795587, upper bound: 1.8800166

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8798288, upper bound: 1.8792508
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8796533, upper bound: 1.8793698
time: 2.03 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8798090, upper bound: 1.8792727
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8796337, upper bound: 1.8794014
time: 2.10 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8798288, upper bound: 1.8792508
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8796533, upper bound: 1.8793698
time: 2.07 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8798090, upper bound: 1.8792727
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8796337, upper bound: 1.8794014
time: 2.55 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8794014, upper bound: 1.8796338
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8792727, upper bound: 1.8798090
time: 2.51 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8793698, upper bound: 1.8796533
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8792508, upper bound: 1.8798288
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8794014, upper bound: 1.8796338
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8792727, upper bound: 1.8798090
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8793698, upper bound: 1.8796533
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8792508, upper bound: 1.8798288
time: 2.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8798288, upper bound: 1.8792508
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8796533, upper bound: 1.8793698
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8798090, upper bound: 1.8792727
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8796337, upper bound: 1.8794014
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8798288, upper bound: 1.8792508
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8796533, upper bound: 1.8793698
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8798090, upper bound: 1.8792727
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8796337, upper bound: 1.8794014
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8794014, upper bound: 1.8796338
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8792727, upper bound: 1.8798090
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8793698, upper bound: 1.8796533
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8792508, upper bound: 1.8798288
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8794014, upper bound: 1.8796338
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8792727, upper bound: 1.8798090
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8793698, upper bound: 1.8796533
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 6, lower bound: -1.8792508, upper bound: 1.8798288

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543520, upper bound: 1.8538645
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543496, upper bound: 1.8538645
time: 2.08 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
time: 1.95 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543425, upper bound: 1.8538730
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543410, upper bound: 1.8538730
time: 2.08 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
time: 1.98 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543520, upper bound: 1.8538645
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543496, upper bound: 1.8538645
time: 2.11 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
time: 3.02 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543425, upper bound: 1.8538730
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8543410, upper bound: 1.8538730
time: 1.99 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
time: 1.98 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
time: 1.92 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543410
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543425
time: 1.94 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
time: 1.93 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543496
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543520
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543410
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543425
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543495
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543520
time: 2.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543520, upper bound: 1.8538645
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543496, upper bound: 1.8538645
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543425, upper bound: 1.8538730
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543410, upper bound: 1.8538730
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543520, upper bound: 1.8538645
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543496, upper bound: 1.8538645
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541781, upper bound: 1.8540117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543425, upper bound: 1.8538730
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8543410, upper bound: 1.8538730
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8541714, upper bound: 1.8540294
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543410
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543425
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543496
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540294, upper bound: 1.8541714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538730, upper bound: 1.8543425
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8540117, upper bound: 1.8541781
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543495
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.12
Output dim: 6, lower bound: -1.8538645, upper bound: 1.8543520

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457293, upper bound: 1.8452417
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456423, upper bound: 1.8452763
time: 2.05 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457288, upper bound: 1.8452417
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456407, upper bound: 1.8452763
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
time: 1.92 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
time: 1.91 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457161, upper bound: 1.8452456
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456349, upper bound: 1.8452876
time: 1.86 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457160, upper bound: 1.8452456
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456343, upper bound: 1.8452876
time: 2.03 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
time: 2.13 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
time: 2.12 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457293, upper bound: 1.8452417
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456423, upper bound: 1.8452763
time: 2.05 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457288, upper bound: 1.8452417
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456407, upper bound: 1.8452763
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
time: 1.79 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457161, upper bound: 1.8452456
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456349, upper bound: 1.8452876
time: 1.95 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457160, upper bound: 1.8452456
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8456343, upper bound: 1.8452876
time: 2.03 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
time: 2.18 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
time: 2.09 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
time: 2.76 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454897
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
time: 2.63 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456343
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457160
time: 2.09 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456349
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457161
time: 2.10 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
time: 1.91 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
time: 1.87 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456407
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457288
time: 2.00 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456423
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457293
time: 1.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
time: 2.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
time: 2.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456343
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457160
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456349
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457161
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456407
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457288
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456423
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457293
time: 1.98 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457293, upper bound: 1.8452417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456423, upper bound: 1.8452763
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457288, upper bound: 1.8452417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456407, upper bound: 1.8452763
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457161, upper bound: 1.8452456
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456349, upper bound: 1.8452876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457160, upper bound: 1.8452456
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456343, upper bound: 1.8452876
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457293, upper bound: 1.8452417
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456423, upper bound: 1.8452763
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457288, upper bound: 1.8452417
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456407, upper bound: 1.8452763
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455419, upper bound: 1.8453685
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454906, upper bound: 1.8454185
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457161, upper bound: 1.8452456
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456349, upper bound: 1.8452876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8457160, upper bound: 1.8452456
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8456343, upper bound: 1.8452876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8455378, upper bound: 1.8453781
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454897, upper bound: 1.8454405
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454897
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456349
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457161
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456407
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457288
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456423
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457293
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454405, upper bound: 1.8454898
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453781, upper bound: 1.8455378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456343
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452876, upper bound: 1.8456349
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452456, upper bound: 1.8457161
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8454185, upper bound: 1.8454906
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8453685, upper bound: 1.8455419
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456407
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457288
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452763, upper bound: 1.8456423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 6, lower bound: -1.8452417, upper bound: 1.8457293

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8409440, upper bound: 1.8404601
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8409440, upper bound: 1.8404601
time: 1.85 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8408718, upper bound: 1.8404970
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8408718, upper bound: 1.8404970
time: 2.03 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8409429, upper bound: 1.8404601
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8409429, upper bound: 1.8404601
time: 2.14 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8408656, upper bound: 1.8404970
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8408656, upper bound: 1.8404970
time: 1.95 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407669, upper bound: 1.8405898
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407669, upper bound: 1.8405898
time: 2.25 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407176, upper bound: 1.8406310
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407176, upper bound: 1.8406310
time: 1.77 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407664, upper bound: 1.8405898
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407664, upper bound: 1.8405898
time: 2.17 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407154, upper bound: 1.8406310
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407154, upper bound: 1.8406310
time: 2.17 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9112678, upper bound: 1.9107672
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9107672, upper bound: 1.9112678
time: 2.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.80
Output dim: 6, lower bound: -1.9112678, upper bound: 1.9107672
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.80
Output dim: 6, lower bound: -1.9107672, upper bound: 1.9112678

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781135, upper bound: 1.8777670
time: 2.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781135, upper bound: 1.8777670
time: 2.74 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8781136
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8781136
time: 2.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.22
Output dim: 6, lower bound: -1.8781135, upper bound: 1.8777670
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.22
Output dim: 6, lower bound: -1.8781135, upper bound: 1.8777670
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.22
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8781136
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.22
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8781136

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781136, upper bound: 1.8777164
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8780651, upper bound: 1.8777670
time: 2.35 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781136, upper bound: 1.8777164
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8780651, upper bound: 1.8777670
time: 2.50 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8780651
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777164, upper bound: 1.8781134
time: 2.34 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8780651
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8777164, upper bound: 1.8781134
time: 2.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8781136, upper bound: 1.8777164
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8780651, upper bound: 1.8777670
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8781136, upper bound: 1.8777164
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8780651, upper bound: 1.8777670
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8780651
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8777164, upper bound: 1.8781134
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8777670, upper bound: 1.8780651
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 6, lower bound: -1.8777164, upper bound: 1.8781134

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8779176, upper bound: 1.8774852
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778608, upper bound: 1.8775297
time: 2.71 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778684, upper bound: 1.8775176
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778060, upper bound: 1.8775721
time: 2.25 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8779176, upper bound: 1.8774852
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778608, upper bound: 1.8775297
time: 2.70 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778684, upper bound: 1.8775176
time: 2.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778060, upper bound: 1.8775721
time: 2.14 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775721, upper bound: 1.8778060
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775176, upper bound: 1.8778684
time: 2.23 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775297, upper bound: 1.8778608
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774852, upper bound: 1.8779176
time: 2.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775721, upper bound: 1.8778060
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775176, upper bound: 1.8778683
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8775297, upper bound: 1.8778608
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774852, upper bound: 1.8779176
time: 3.27 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8779176, upper bound: 1.8774852
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778608, upper bound: 1.8775297
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778684, upper bound: 1.8775176
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778060, upper bound: 1.8775721
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8779176, upper bound: 1.8774852
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778608, upper bound: 1.8775297
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778684, upper bound: 1.8775176
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8778060, upper bound: 1.8775721
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775721, upper bound: 1.8778060
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775176, upper bound: 1.8778684
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775297, upper bound: 1.8778608
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8774852, upper bound: 1.8779176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775721, upper bound: 1.8778060
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775176, upper bound: 1.8778683
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8775297, upper bound: 1.8778608
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 6, lower bound: -1.8774852, upper bound: 1.8779176

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
time: 2.21 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
time: 2.19 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
time: 2.47 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
time: 2.27 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
time: 2.09 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
time: 2.20 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
time: 2.08 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
time: 2.27 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
time: 2.31 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
time: 2.00 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
time: 2.15 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
time: 11.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
time: 2.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
time: 2.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
time: 12.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
time: 5.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8526200, upper bound: 1.8522643
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525586, upper bound: 1.8523286
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525959, upper bound: 1.8522891
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8525338, upper bound: 1.8523530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523530, upper bound: 1.8525338
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522891, upper bound: 1.8525959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8523286, upper bound: 1.8525587
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 6, lower bound: -1.8522643, upper bound: 1.8526200

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438414
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
time: 2.13 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
time: 2.14 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
time: 2.33 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
time: 2.34 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
time: 2.57 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
time: 2.46 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
time: 2.48 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439277
time: 2.34 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
time: 2.12 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
time: 2.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
time: 2.26 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
time: 2.58 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
time: 2.43 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
time: 2.26 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
time: 2.28 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
time: 2.26 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
time: 2.23 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
time: 2.18 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
time: 2.15 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441092
time: 2.05 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441093
time: 2.06 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
time: 2.34 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
time: 2.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
time: 2.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
time: 2.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441092
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441093
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441224
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
time: 2.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438414
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441668, upper bound: 1.8438413
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441222, upper bound: 1.8438515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441092, upper bound: 1.8438935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440692, upper bound: 1.8439114
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441468, upper bound: 1.8438526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8441113, upper bound: 1.8438691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440863, upper bound: 1.8439041
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8440603, upper bound: 1.8439278
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441092
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441093
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439277, upper bound: 1.8440603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439041, upper bound: 1.8440863
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438692, upper bound: 1.8441113
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438526, upper bound: 1.8441468
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8439114, upper bound: 1.8440692
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438935, upper bound: 1.8441093
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441222
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438515, upper bound: 1.8441224
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.71
Output dim: 6, lower bound: -1.8438414, upper bound: 1.8441668

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8395239, upper bound: 1.8391998
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8395239, upper bound: 1.8391997
time: 2.49 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9065810, upper bound: 1.9060745
time: 3.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9060745, upper bound: 1.9065810
time: 3.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.28
Output dim: 6, lower bound: -1.9065810, upper bound: 1.9060745
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.28
Output dim: 6, lower bound: -1.9060745, upper bound: 1.9065810

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8743759, upper bound: 1.8740390
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8743759, upper bound: 1.8740390
time: 2.66 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8740390, upper bound: 1.8743759
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8740390, upper bound: 1.8743759
time: 2.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 6, lower bound: -1.8743759, upper bound: 1.8740390
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 6, lower bound: -1.8743759, upper bound: 1.8740390
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 6, lower bound: -1.8740390, upper bound: 1.8743759
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 6, lower bound: -1.8740390, upper bound: 1.8743759

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8743503, upper bound: 1.8739187
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742081, upper bound: 1.8740069
time: 2.91 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8743503, upper bound: 1.8739187
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742081, upper bound: 1.8740069
time: 2.78 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8740069, upper bound: 1.8742081
time: 3.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739187, upper bound: 1.8743503
time: 3.03 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8740069, upper bound: 1.8742081
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739187, upper bound: 1.8743503
time: 3.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8743503, upper bound: 1.8739187
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8742081, upper bound: 1.8740069
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8743503, upper bound: 1.8739187
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8742081, upper bound: 1.8740069
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8740069, upper bound: 1.8742081
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8739187, upper bound: 1.8743503
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8740069, upper bound: 1.8742081
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.40
Output dim: 6, lower bound: -1.8739187, upper bound: 1.8743503

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742617, upper bound: 1.8738314
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742620, upper bound: 1.8738317
time: 2.98 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8741152, upper bound: 1.8739204
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8741143, upper bound: 1.8739200
time: 2.58 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742617, upper bound: 1.8738314
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742620, upper bound: 1.8738317
time: 3.03 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8741152, upper bound: 1.8739204
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8741143, upper bound: 1.8739200
time: 2.58 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739200, upper bound: 1.8741143
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739204, upper bound: 1.8741152
time: 2.60 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738317, upper bound: 1.8742620
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738314, upper bound: 1.8742617
time: 2.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739200, upper bound: 1.8741143
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739204, upper bound: 1.8741152
time: 2.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738317, upper bound: 1.8742620
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738314, upper bound: 1.8742617
time: 2.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8742617, upper bound: 1.8738314
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8742620, upper bound: 1.8738317
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8741152, upper bound: 1.8739204
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8741143, upper bound: 1.8739200
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8742617, upper bound: 1.8738314
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8742620, upper bound: 1.8738317
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8741152, upper bound: 1.8739204
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8741143, upper bound: 1.8739200
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8739200, upper bound: 1.8741143
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8739204, upper bound: 1.8741152
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8738317, upper bound: 1.8742620
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8738314, upper bound: 1.8742617
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8739200, upper bound: 1.8741143
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8739204, upper bound: 1.8741152
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8738317, upper bound: 1.8742620
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.90
Output dim: 6, lower bound: -1.8738314, upper bound: 1.8742617

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
time: 2.78 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
time: 2.38 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
time: 3.20 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
time: 2.93 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
time: 3.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
time: 2.88 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
time: 2.41 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
time: 3.16 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
time: 2.67 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
time: 2.60 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
time: 2.53 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
time: 2.95 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
time: 2.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
time: 2.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
time: 2.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
time: 3.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
time: 2.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8496098, upper bound: 1.8493286
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495959, upper bound: 1.8493342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495404, upper bound: 1.8494041
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8495246, upper bound: 1.8494129
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494129, upper bound: 1.8495246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8494041, upper bound: 1.8495404
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493342, upper bound: 1.8495958
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.95
Output dim: 6, lower bound: -1.8493286, upper bound: 1.8496098

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412579, upper bound: 1.8410476
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412426, upper bound: 1.8410499
time: 3.65 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412579, upper bound: 1.8410476
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412426, upper bound: 1.8410499
time: 2.99 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412492, upper bound: 1.8410524
time: 3.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412380, upper bound: 1.8410545
time: 3.16 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412492, upper bound: 1.8410524
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412380, upper bound: 1.8410546
time: 2.81 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412207, upper bound: 1.8410727
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412109, upper bound: 1.8410793
time: 3.11 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412207, upper bound: 1.8410727
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412109, upper bound: 1.8410793
time: 3.40 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412115, upper bound: 1.8410783
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412045, upper bound: 1.8410863
time: 3.33 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412115, upper bound: 1.8410783
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412045, upper bound: 1.8410863
time: 3.25 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412579, upper bound: 1.8410476
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412426, upper bound: 1.8410499
time: 4.04 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412579, upper bound: 1.8410476
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412426, upper bound: 1.8410499
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412492, upper bound: 1.8410524
time: 3.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412380, upper bound: 1.8410546
time: 3.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412492, upper bound: 1.8410524
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412380, upper bound: 1.8410546
time: 2.88 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412207, upper bound: 1.8410726
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412109, upper bound: 1.8410793
time: 3.13 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412207, upper bound: 1.8410727
time: 3.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412109, upper bound: 1.8410793
time: 3.38 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412115, upper bound: 1.8410783
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412045, upper bound: 1.8410863
time: 3.26 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412115, upper bound: 1.8410783
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412045, upper bound: 1.8410863
time: 3.33 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410863, upper bound: 1.8412045
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410783, upper bound: 1.8412115
time: 2.81 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410863, upper bound: 1.8412045
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410783, upper bound: 1.8412115
time: 2.88 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410793, upper bound: 1.8412109
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410727, upper bound: 1.8412207
time: 2.98 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410793, upper bound: 1.8412109
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410727, upper bound: 1.8412207
time: 2.74 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410546, upper bound: 1.8412380
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410524, upper bound: 1.8412492
time: 3.44 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410546, upper bound: 1.8412380
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410524, upper bound: 1.8412492
time: 3.14 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410499, upper bound: 1.8412426
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410476, upper bound: 1.8412579
time: 2.59 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410499, upper bound: 1.8412426
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410476, upper bound: 1.8412579
time: 3.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410863, upper bound: 1.8412045
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410783, upper bound: 1.8412115
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410863, upper bound: 1.8412045
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410783, upper bound: 1.8412115
time: 3.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410793, upper bound: 1.8412109
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410727, upper bound: 1.8412207
time: 3.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410793, upper bound: 1.8412109
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410727, upper bound: 1.8412207
time: 2.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410546, upper bound: 1.8412380
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8410524, upper bound: 1.8412492
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.094357967376709
rel_dist={6: [-1.9066430217684205, 1.9066430217684207]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1803.50 seconds
