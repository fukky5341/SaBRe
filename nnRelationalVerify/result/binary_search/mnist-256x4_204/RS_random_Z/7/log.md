## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 81.1446251145
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393)
1: (-39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019)
2: (-53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567)
3: (-59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770)
4: (-57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253)
5: (-50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782)
6: (-52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593)
7: (-47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050)
8: (-63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007)
9: (-44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377)

## BASE Result
execution time: IAR + LP analysis = 1.34 + 8.33 = 9.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -81.2973621, upper bound: 81.2973621


# Binary Search by BASE starts (time budget: 2690.33 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=90.90605926513672
rel_dist={6: [-81.29714583279736, 81.29714583279738]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=90.90605926513672
rel_dist={6: [-81.29696620335748, 81.29696620339547]}

## Binary Search Result
Binary search time: 38.06 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2652.27 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
time: 7.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.44
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.44
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
time: 8.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2501289, upper bound: 81.2501289
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2501289, upper bound: 81.2501289
time: 7.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 6, lower bound: -81.2800410, upper bound: 81.2800410
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 6, lower bound: -81.2501289, upper bound: 81.2501289
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 6, lower bound: -81.2501289, upper bound: 81.2501289

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2739030, upper bound: 81.2739030
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2739030, upper bound: 81.2739030
time: 8.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2660935, upper bound: 81.2660935
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2660935, upper bound: 81.2660935
time: 8.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743180
time: 8.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743180
time: 8.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2465522, upper bound: 81.2465522
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2465522, upper bound: 81.2465522
time: 8.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2739030, upper bound: 81.2739030
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2739030, upper bound: 81.2739030
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2660935, upper bound: 81.2660935
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2660935, upper bound: 81.2660935
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743180
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2465522, upper bound: 81.2465522
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.84
Output dim: 6, lower bound: -81.2465522, upper bound: 81.2465522

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
time: 8.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461865
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461865
time: 9.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2466505, upper bound: 81.2466530
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2466505, upper bound: 81.2466530
time: 10.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2575575, upper bound: 81.2575750
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2575750, upper bound: 81.2575575
time: 9.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743173, upper bound: 81.1743180
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743173
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739934
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1739934, upper bound: 81.1739936
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
time: 8.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2365854, upper bound: 81.2365873
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2365854, upper bound: 81.2365854
time: 7.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461865
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461865
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2466505, upper bound: 81.2466530
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2466505, upper bound: 81.2466530
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2575575, upper bound: 81.2575750
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2575750, upper bound: 81.2575575
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.1743173, upper bound: 81.1743180
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743173
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739934
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.1739934, upper bound: 81.1739936
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2365854, upper bound: 81.2365873
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.91
Output dim: 6, lower bound: -81.2365854, upper bound: 81.2365854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690257
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
time: 9.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690150, upper bound: 81.2690255
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690150
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2461839, upper bound: 81.2461865
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461841
time: 7.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2366107, upper bound: 81.2366270
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2366270, upper bound: 81.2366104
time: 8.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2026435, upper bound: 81.2026419
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2026435, upper bound: 81.2026419
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2465015, upper bound: 81.2465099
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2465073, upper bound: 81.2465059
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2563199, upper bound: 81.2563449
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2563291, upper bound: 81.2563392
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2575750, upper bound: 81.2575541
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2575707, upper bound: 81.2575575
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743156, upper bound: 81.1743180
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743173, upper bound: 81.1743177
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743160
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743174, upper bound: 81.1743173
time: 6.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739934
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739929
time: 9.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1673639, upper bound: 81.1673654
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1673639, upper bound: 81.1673654
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425108
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425111, upper bound: 81.2425232
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
time: 9.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2254584, upper bound: 81.2255018
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2254584, upper bound: 81.2255018
time: 8.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2365874, upper bound: 81.2365854
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2365874, upper bound: 81.2365854
time: 6.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690257
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690255
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2690150, upper bound: 81.2690255
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2690257, upper bound: 81.2690150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2461839, upper bound: 81.2461865
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2461872, upper bound: 81.2461841
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2366107, upper bound: 81.2366270
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2366270, upper bound: 81.2366104
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2026435, upper bound: 81.2026419
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2026435, upper bound: 81.2026419
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2465015, upper bound: 81.2465099
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2465073, upper bound: 81.2465059
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2563199, upper bound: 81.2563449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2563291, upper bound: 81.2563392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2575750, upper bound: 81.2575541
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2575707, upper bound: 81.2575575
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1743156, upper bound: 81.1743180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1743173, upper bound: 81.1743177
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1743180, upper bound: 81.1743160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1743174, upper bound: 81.1743173
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739934
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1739936, upper bound: 81.1739929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1673639, upper bound: 81.1673654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.1673639, upper bound: 81.1673654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425108
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2425111, upper bound: 81.2425232
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2425232, upper bound: 81.2425232
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2254584, upper bound: 81.2255018
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2254584, upper bound: 81.2255018
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2365874, upper bound: 81.2365854
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.93
Output dim: 6, lower bound: -81.2365874, upper bound: 81.2365854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2679936, upper bound: 81.2679965
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2679965, upper bound: 81.2679936
time: 8.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2687270, upper bound: 81.2687297
time: 12.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2687299, upper bound: 81.2687270
time: 7.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2157049, upper bound: 81.2157049
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2157049, upper bound: 81.2157049
time: 10.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.25
Output dim: 6, lower bound: -81.2157049, upper bound: 81.2157049
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.25
Output dim: 6, lower bound: -81.2157049, upper bound: 81.2157049

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104214
time: 9.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104208
time: 7.94 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137755, upper bound: 81.2137754
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
time: 12.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104214
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104208
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 6, lower bound: -81.2137755, upper bound: 81.2137754
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104206, upper bound: 81.2104214
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104213
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104207
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104208
time: 7.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137755, upper bound: 81.2137750
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
time: 11.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
time: 11.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2104206, upper bound: 81.2104214
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104213
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104207
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104208
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2137755, upper bound: 81.2137750
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.39
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044330, upper bound: 81.2044312
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044309, upper bound: 81.2044330
time: 9.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104213
time: 10.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
time: 9.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
time: 7.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104208
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104206
time: 9.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
time: 8.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2098875, upper bound: 81.2098766
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2098875, upper bound: 81.2098766
time: 6.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137748, upper bound: 81.2137755
time: 10.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
time: 7.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137753, upper bound: 81.2137755
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
time: 10.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2044330, upper bound: 81.2044312
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2044309, upper bound: 81.2044330
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104213
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104208
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2104214, upper bound: 81.2104206
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2098875, upper bound: 81.2098766
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2098875, upper bound: 81.2098766
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2137748, upper bound: 81.2137755
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2137753, upper bound: 81.2137755
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1925559, upper bound: 81.1925610
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1925578, upper bound: 81.1925590
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1841755, upper bound: 81.1841829
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1841755, upper bound: 81.1841829
time: 6.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
time: 7.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104206
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104204, upper bound: 81.2104213
time: 7.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2023185, upper bound: 81.2023186
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2023185, upper bound: 81.2023186
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044095, upper bound: 81.2044081
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
time: 7.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104208
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2104212, upper bound: 81.2104203
time: 9.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2084310, upper bound: 81.2084299
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2084306, upper bound: 81.2084294
time: 7.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1619641, upper bound: 81.1619638
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1619641, upper bound: 81.1619638
time: 7.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
time: 9.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1663115, upper bound: 81.1663118
time: 12.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1663115, upper bound: 81.1663118
time: 18.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1967152, upper bound: 81.1967142
time: 11.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1967141, upper bound: 81.1967156
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1935423, upper bound: 81.1935453
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1935423, upper bound: 81.1935453
time: 7.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137746
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
time: 8.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1925559, upper bound: 81.1925610
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1925578, upper bound: 81.1925590
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1841755, upper bound: 81.1841829
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1841755, upper bound: 81.1841829
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104208, upper bound: 81.2104206
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104204, upper bound: 81.2104213
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2023185, upper bound: 81.2023186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2023185, upper bound: 81.2023186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2044095, upper bound: 81.2044081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2044096, upper bound: 81.2044081
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104213, upper bound: 81.2104208
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2104212, upper bound: 81.2104203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2084310, upper bound: 81.2084299
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2084306, upper bound: 81.2084294
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1619641, upper bound: 81.1619638
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1619641, upper bound: 81.1619638
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1696631, upper bound: 81.1696571
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1663115, upper bound: 81.1663118
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1663115, upper bound: 81.1663118
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1967152, upper bound: 81.1967142
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1967141, upper bound: 81.1967156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1935423, upper bound: 81.1935453
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.1935423, upper bound: 81.1935453
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.26
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137754
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.26
Output dim: 6, lower bound: -81.2137753, upper bound: 81.2137755
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.26
Output dim: 6, lower bound: -81.2137754, upper bound: 81.2137755
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=90.90605926513672
rel_dist={6: [-81.29714583279736, 81.29714583279738]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2859589, upper bound: 81.2859589
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2859589, upper bound: 81.2859589
time: 9.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.35
Output dim: 6, lower bound: -81.2859589, upper bound: 81.2859589
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.35
Output dim: 6, lower bound: -81.2859589, upper bound: 81.2859589

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844116, upper bound: 81.2844116
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844116, upper bound: 81.2844112
time: 9.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2658300, upper bound: 81.2658300
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2658300, upper bound: 81.2658300
time: 9.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.82
Output dim: 6, lower bound: -81.2844116, upper bound: 81.2844116
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.82
Output dim: 6, lower bound: -81.2844116, upper bound: 81.2844112
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.82
Output dim: 6, lower bound: -81.2658300, upper bound: 81.2658300
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.82
Output dim: 6, lower bound: -81.2658300, upper bound: 81.2658300

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811274
time: 11.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811274
time: 10.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797666, upper bound: 81.2797690
time: 10.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797691, upper bound: 81.2797666
time: 8.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323825
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323825
time: 17.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2639206, upper bound: 81.2639206
time: 10.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2639206, upper bound: 81.2639206
time: 10.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811274
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811274
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2797666, upper bound: 81.2797690
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2797691, upper bound: 81.2797666
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323825
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323825
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2639206, upper bound: 81.2639206
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 6, lower bound: -81.2639206, upper bound: 81.2639206

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811271
time: 10.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811271, upper bound: 81.2811274
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2732355, upper bound: 81.2732355
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2732355, upper bound: 81.2732355
time: 10.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797667, upper bound: 81.2797650
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797636, upper bound: 81.2797690
time: 8.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692086
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692086
time: 7.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034780
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034780
time: 10.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323817
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323825
time: 8.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2050567, upper bound: 81.2050564
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2050567, upper bound: 81.2050567
time: 9.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2584647, upper bound: 81.2584676
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2584676, upper bound: 81.2584647
time: 9.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2811274, upper bound: 81.2811271
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2811271, upper bound: 81.2811274
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2732355, upper bound: 81.2732355
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2732355, upper bound: 81.2732355
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2797667, upper bound: 81.2797650
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2797636, upper bound: 81.2797690
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692086
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692086
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034780
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034780
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323817
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323825
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2050567, upper bound: 81.2050564
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2050567, upper bound: 81.2050567
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2584647, upper bound: 81.2584676
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.55
Output dim: 6, lower bound: -81.2584676, upper bound: 81.2584647

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2641223, upper bound: 81.2641242
time: 9.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2641223, upper bound: 81.2641242
time: 8.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811271, upper bound: 81.2811271
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2811269, upper bound: 81.2811274
time: 7.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2732352, upper bound: 81.2732352
time: 9.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2732352, upper bound: 81.2732355
time: 10.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2618589, upper bound: 81.2618589
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2618589, upper bound: 81.2618589
time: 8.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797633, upper bound: 81.2797650
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797666, upper bound: 81.2797640
time: 9.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797630, upper bound: 81.2797690
time: 10.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2797635, upper bound: 81.2797684
time: 8.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2655030, upper bound: 81.2655030
time: 11.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2655030, upper bound: 81.2655030
time: 10.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2692080, upper bound: 81.2692086
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692079
time: 8.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1975006, upper bound: 81.1975012
time: 8.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1975012, upper bound: 81.1975006
time: 8.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034779
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2034779, upper bound: 81.2034780
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323824, upper bound: 81.2323816
time: 12.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323817
time: 9.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323825
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2323816, upper bound: 81.2323824
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393
1: -39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019
2: -53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567
3: -59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770
4: -57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253
5: -50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782
6: -52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593
7: -47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050
8: -63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007
9: -44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1789972, upper bound: 81.1789972
time: 10.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1789972, upper bound: 81.1789972
time: 8.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2641223, upper bound: 81.2641242
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2641223, upper bound: 81.2641242
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2811271, upper bound: 81.2811271
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2811269, upper bound: 81.2811274
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2732352, upper bound: 81.2732352
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2732352, upper bound: 81.2732355
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2618589, upper bound: 81.2618589
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2618589, upper bound: 81.2618589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2797633, upper bound: 81.2797650
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2797666, upper bound: 81.2797640
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2797630, upper bound: 81.2797690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2797635, upper bound: 81.2797684
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2655030, upper bound: 81.2655030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2655030, upper bound: 81.2655030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2692080, upper bound: 81.2692086
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2692086, upper bound: 81.2692079
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.1975006, upper bound: 81.1975012
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.1975012, upper bound: 81.1975006
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2034780, upper bound: 81.2034779
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2034779, upper bound: 81.2034780
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2323824, upper bound: 81.2323816
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2323825, upper bound: 81.2323817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2323818, upper bound: 81.2323825
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.2323816, upper bound: 81.2323824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.1789972, upper bound: 81.1789972
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.35
Output dim: 6, lower bound: -81.1789972, upper bound: 81.1789972
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.35
Output dim: 6, lower bound: -81.2050567, upper bound: 81.2050567
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.35
Output dim: 6, lower bound: -81.2584647, upper bound: 81.2584676
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.35
Output dim: 6, lower bound: -81.2584676, upper bound: 81.2584647
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=90.90605926513672
rel_dist={6: [-81.29696620335748, 81.29696620339547]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1818.87 seconds
