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
execution time: IAR + LP analysis = 1.55 + 8.85 = 10.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -81.2973621, upper bound: 81.2973621


# Binary Search by BASE starts (time budget: 2689.60 seconds, max iter: 100)

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
Binary search time: 38.25 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2651.35 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935164, upper bound: 81.2935216
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935216, upper bound: 81.2935164
time: 6.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.36
Output dim: 6, lower bound: -81.2935164, upper bound: 81.2935216
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.36
Output dim: 6, lower bound: -81.2935216, upper bound: 81.2935164

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2846752, upper bound: 81.2847022
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2846751, upper bound: 81.2847023
time: 10.88 seconds

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
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847023, upper bound: 81.2846751
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847023, upper bound: 81.2846752
time: 8.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 6, lower bound: -81.2846752, upper bound: 81.2847022
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 6, lower bound: -81.2846751, upper bound: 81.2847023
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 6, lower bound: -81.2847023, upper bound: 81.2846751
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 6, lower bound: -81.2847023, upper bound: 81.2846752

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
time: 6.28 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
time: 5.83 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
time: 5.31 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
time: 5.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592947, upper bound: 81.1592921
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.28
Output dim: 6, lower bound: -81.1592921, upper bound: 81.1592947

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
time: 6.85 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
time: 6.86 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
time: 6.89 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
time: 6.85 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
time: 6.44 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
time: 6.40 seconds

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
time: 5.60 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
time: 5.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592867, upper bound: 81.1592828
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592853, upper bound: 81.1592839
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592841, upper bound: 81.1592852
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -81.1592828, upper bound: 81.1592867

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 5.75 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 5.80 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 5.75 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 5.81 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 5.36 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 5.76 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
time: 5.37 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
time: 5.77 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 5.52 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 5.71 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 5.52 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 5.70 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 5.55 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 5.63 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
time: 5.55 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
time: 5.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446179
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446194
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446194, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.69
Output dim: 6, lower bound: -81.1446179, upper bound: 81.1446141
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935434, upper bound: 81.2935505
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935505, upper bound: 81.2935434
time: 8.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.47
Output dim: 6, lower bound: -81.2935434, upper bound: 81.2935505
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.47
Output dim: 6, lower bound: -81.2935505, upper bound: 81.2935434

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2846993, upper bound: 81.2847320
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2846993, upper bound: 81.2847320
time: 6.60 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847320, upper bound: 81.2846993
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847320, upper bound: 81.2846993
time: 6.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.15
Output dim: 6, lower bound: -81.2846993, upper bound: 81.2847320
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.15
Output dim: 6, lower bound: -81.2846993, upper bound: 81.2847320
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.15
Output dim: 6, lower bound: -81.2847320, upper bound: 81.2846993
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.15
Output dim: 6, lower bound: -81.2847320, upper bound: 81.2846993

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
time: 7.43 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
time: 7.44 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
time: 6.25 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
time: 6.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592999, upper bound: 81.1592961
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.98
Output dim: 6, lower bound: -81.1592961, upper bound: 81.1592999

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
time: 5.23 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
time: 5.14 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
time: 5.16 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
time: 5.15 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592897
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
time: 6.20 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592897
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
time: 6.22 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592899
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
time: 6.22 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592899
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
time: 6.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592861
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592899, upper bound: 81.1592878
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592897
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592897
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592899
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592880, upper bound: 81.1592899
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.13
Output dim: 6, lower bound: -81.1592861, upper bound: 81.1592918

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 5.92 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 5.92 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 5.93 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 5.96 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 5.91 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 6.36 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
time: 5.93 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
time: 6.36 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 6.07 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.31 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 6.04 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.31 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 6.06 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.69 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
time: 6.05 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
time: 5.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446192
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446211
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446211, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.72
Output dim: 6, lower bound: -81.1446192, upper bound: 81.1446141
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=90.90605926513672
rel_dist={6: [-81.2973378458658, 81.2973378458658]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935610, upper bound: 81.2935686
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935686, upper bound: 81.2935610
time: 5.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.57
Output dim: 6, lower bound: -81.2935610, upper bound: 81.2935686
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.57
Output dim: 6, lower bound: -81.2935686, upper bound: 81.2935610

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847142, upper bound: 81.2847506
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847142, upper bound: 81.2847506
time: 6.57 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847506, upper bound: 81.2847142
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847506, upper bound: 81.2847142
time: 6.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.96
Output dim: 6, lower bound: -81.2847142, upper bound: 81.2847506
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.96
Output dim: 6, lower bound: -81.2847142, upper bound: 81.2847506
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.96
Output dim: 6, lower bound: -81.2847506, upper bound: 81.2847142
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.96
Output dim: 6, lower bound: -81.2847506, upper bound: 81.2847142

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
time: 5.66 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
time: 6.35 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
time: 13.54 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
time: 10.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
time: 10.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1593034, upper bound: 81.1592988
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.49
Output dim: 6, lower bound: -81.1592988, upper bound: 81.1593034

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
time: 10.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
time: 5.26 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
time: 10.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
time: 5.26 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
time: 10.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
time: 5.26 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
time: 10.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
time: 5.27 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
time: 5.42 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
time: 5.44 seconds

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
time: 4.65 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
time: 4.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592952, upper bound: 81.1592882
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592929, upper bound: 81.1592904
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592905, upper bound: 81.1592927
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.82
Output dim: 6, lower bound: -81.1592883, upper bound: 81.1592951

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.85 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.78 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.86 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.18 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.86 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.82 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 4.84 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 6.83 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.74 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.23 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.73 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.28 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.16 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.25 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.15 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.25 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=90.90605926513672
rel_dist={6: [-81.29735402320546, 81.29735402320546]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935696, upper bound: 81.2935775
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2935775, upper bound: 81.2935696
time: 6.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 6, lower bound: -81.2935696, upper bound: 81.2935775
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 6, lower bound: -81.2935775, upper bound: 81.2935696

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847214, upper bound: 81.2847593
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847214, upper bound: 81.2847593
time: 6.90 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847593, upper bound: 81.2847214
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2847593, upper bound: 81.2847214
time: 6.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 6, lower bound: -81.2847214, upper bound: 81.2847593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 6, lower bound: -81.2847214, upper bound: 81.2847593
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 6, lower bound: -81.2847593, upper bound: 81.2847214
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 6, lower bound: -81.2847593, upper bound: 81.2847214

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
time: 5.54 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
time: 5.52 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
time: 6.74 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
time: 7.26 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593051, upper bound: 81.1593001
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.75
Output dim: 6, lower bound: -81.1593001, upper bound: 81.1593051

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
time: 4.90 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
time: 4.87 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
time: 5.43 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
time: 5.43 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
time: 4.82 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
time: 4.80 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
time: 5.14 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
time: 5.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592969, upper bound: 81.1592892
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592944, upper bound: 81.1592917
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592918, upper bound: 81.1592942
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 6, lower bound: -81.1592894, upper bound: 81.1592968

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 5.11 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.71 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 5.55 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.71 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 5.11 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.73 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
time: 5.14 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
time: 5.70 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.98 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.34 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 5.71 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.06 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.10 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 7.18 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
time: 6.26 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
time: 6.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446141, upper bound: 81.1446218
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446218, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.01
Output dim: 6, lower bound: -81.1446196, upper bound: 81.1446141
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=90.90605926513672
rel_dist={6: [-81.29736206815559, 81.29736206815559]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 1780.50 seconds
