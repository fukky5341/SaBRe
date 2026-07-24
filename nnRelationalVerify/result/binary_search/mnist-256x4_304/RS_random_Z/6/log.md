## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 107.2381207338
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414)
1: (-55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092)
2: (-70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666)
3: (-81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354)
4: (-72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363)
5: (-62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670)
6: (-60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196)
7: (-69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519)
8: (-77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454)
9: (-60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618)

## BASE Result
execution time: IAR + LP analysis = 1.36 + 12.04 = 13.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3456123, upper bound: 107.3456123


# Binary Search by BASE starts (time budget: 1986.60 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search Result
Binary search time: 44.58 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1942.02 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014617
time: 12.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014617
time: 15.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 28.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 28.47
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014617
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 28.47
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014617

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014600, upper bound: 107.3014617
time: 7.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2987746, upper bound: 107.2987746
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2987746, upper bound: 107.2987746
time: 7.17 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 7, lower bound: -107.3014600, upper bound: 107.3014617
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 7, lower bound: -107.2987746, upper bound: 107.2987746
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 7, lower bound: -107.2987746, upper bound: 107.2987746

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014616, upper bound: 107.3014600
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974131, upper bound: 107.2974136
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974151
time: 9.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2956574, upper bound: 107.2956523
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2956574, upper bound: 107.2956523
time: 7.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2919769, upper bound: 107.2919848
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2919769, upper bound: 107.2919848
time: 7.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.3014616, upper bound: 107.3014600
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2974131, upper bound: 107.2974136
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974151
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2956574, upper bound: 107.2956523
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2956574, upper bound: 107.2956523
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2919769, upper bound: 107.2919848
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.90
Output dim: 7, lower bound: -107.2919769, upper bound: 107.2919848

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906281
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906282
time: 8.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014611, upper bound: 107.3014600
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974131, upper bound: 107.2974129
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974128, upper bound: 107.2974136
time: 7.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974147
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974151
time: 9.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2886754, upper bound: 107.2886695
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2886754, upper bound: 107.2886695
time: 6.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910654, upper bound: 107.2910640
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910654, upper bound: 107.2910640
time: 7.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907592, upper bound: 107.2907611
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907593, upper bound: 107.2907642
time: 6.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2882456, upper bound: 107.2882459
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2882456, upper bound: 107.2882459
time: 6.94 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906281
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906282
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.3014611, upper bound: 107.3014600
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.3014617, upper bound: 107.3014600
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2974131, upper bound: 107.2974129
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2974128, upper bound: 107.2974136
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974147
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974151
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2886754, upper bound: 107.2886695
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2886754, upper bound: 107.2886695
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2910654, upper bound: 107.2910640
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2910654, upper bound: 107.2910640
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2907592, upper bound: 107.2907611
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2907593, upper bound: 107.2907642
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2882456, upper bound: 107.2882459
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.07
Output dim: 7, lower bound: -107.2882456, upper bound: 107.2882459

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2906246, upper bound: 107.2906282
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906248
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2880371, upper bound: 107.2880350
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2880371, upper bound: 107.2880350
time: 7.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3011860, upper bound: 107.3011825
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3011821, upper bound: 107.3011850
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3004093, upper bound: 107.3004074
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3004093, upper bound: 107.3004073
time: 11.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2961302, upper bound: 107.2961288
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2961303, upper bound: 107.2961303
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974123, upper bound: 107.2974126
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974128, upper bound: 107.2974136
time: 8.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970984, upper bound: 107.2971023
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970976, upper bound: 107.2971021
time: 6.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974143
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974109, upper bound: 107.2974151
time: 9.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2874476, upper bound: 107.2874441
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2874480, upper bound: 107.2874432
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2854572, upper bound: 107.2854541
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2854572, upper bound: 107.2854541
time: 7.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910640
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910635
time: 8.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910630
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2910648, upper bound: 107.2910641
time: 7.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
time: 7.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2845922, upper bound: 107.2845933
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2845922, upper bound: 107.2845933
time: 9.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2906246, upper bound: 107.2906282
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2906261, upper bound: 107.2906248
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2880371, upper bound: 107.2880350
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2880371, upper bound: 107.2880350
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.3011860, upper bound: 107.3011825
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.3011821, upper bound: 107.3011850
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.3004093, upper bound: 107.3004074
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.3004093, upper bound: 107.3004073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2961302, upper bound: 107.2961288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2961303, upper bound: 107.2961303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2974123, upper bound: 107.2974126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2974128, upper bound: 107.2974136
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2970984, upper bound: 107.2971023
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2970976, upper bound: 107.2971021
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2974110, upper bound: 107.2974143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2974109, upper bound: 107.2974151
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2874476, upper bound: 107.2874441
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2874480, upper bound: 107.2874432
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2854572, upper bound: 107.2854541
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2854572, upper bound: 107.2854541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910640
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910635
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2910653, upper bound: 107.2910630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2910648, upper bound: 107.2910641
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2810785, upper bound: 107.2810781
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2845922, upper bound: 107.2845933
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.22
Output dim: 7, lower bound: -107.2845922, upper bound: 107.2845933
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.22
Output dim: 7, lower bound: -107.2882456, upper bound: 107.2882459
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3454661, upper bound: 107.3454660
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3454660, upper bound: 107.3454662
time: 8.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.28
Output dim: 7, lower bound: -107.3454661, upper bound: 107.3454660
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.28
Output dim: 7, lower bound: -107.3454660, upper bound: 107.3454662

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423749, upper bound: 107.3423750
time: 10.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423750, upper bound: 107.3423750
time: 10.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3426318, upper bound: 107.3426317
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3426318, upper bound: 107.3426317
time: 9.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.68
Output dim: 7, lower bound: -107.3423749, upper bound: 107.3423750
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.68
Output dim: 7, lower bound: -107.3423750, upper bound: 107.3423750
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.68
Output dim: 7, lower bound: -107.3426318, upper bound: 107.3426317
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.68
Output dim: 7, lower bound: -107.3426318, upper bound: 107.3426317

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3396375, upper bound: 107.3396376
time: 10.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3396372, upper bound: 107.3396379
time: 11.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3133347, upper bound: 107.3133344
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3133347, upper bound: 107.3133344
time: 9.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399894
time: 10.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399894
time: 12.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105439
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105439
time: 8.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3396375, upper bound: 107.3396376
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3396372, upper bound: 107.3396379
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3133347, upper bound: 107.3133344
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3133347, upper bound: 107.3133344
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399894
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399894
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105439
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.55
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105439

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3299348, upper bound: 107.3299327
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3299348, upper bound: 107.3299327
time: 11.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3231867, upper bound: 107.3231844
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3231867, upper bound: 107.3231844
time: 8.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968753
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968753
time: 10.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3092375, upper bound: 107.3092360
time: 11.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3092360, upper bound: 107.3092360
time: 11.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3398944, upper bound: 107.3398890
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3398890, upper bound: 107.3398942
time: 8.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3399881, upper bound: 107.3399894
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399881
time: 9.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
time: 8.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3105398, upper bound: 107.3105439
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105414
time: 10.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3299348, upper bound: 107.3299327
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3299348, upper bound: 107.3299327
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3231867, upper bound: 107.3231844
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3231867, upper bound: 107.3231844
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968753
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968753
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3092375, upper bound: 107.3092360
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3092360, upper bound: 107.3092360
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3398944, upper bound: 107.3398890
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3398890, upper bound: 107.3398942
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3399881, upper bound: 107.3399894
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3399892, upper bound: 107.3399881
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3105398, upper bound: 107.3105439
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105414

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3295159, upper bound: 107.3295125
time: 11.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3295159, upper bound: 107.3295125
time: 9.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3172218, upper bound: 107.3172230
time: 10.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3172218, upper bound: 107.3172230
time: 9.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2837897, upper bound: 107.2837890
time: 9.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2837897, upper bound: 107.2837890
time: 9.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3231855, upper bound: 107.3231839
time: 11.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3231860, upper bound: 107.3231835
time: 8.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2856335, upper bound: 107.2856328
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2856335, upper bound: 107.2856328
time: 8.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968749
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968743, upper bound: 107.2968753
time: 8.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3087366, upper bound: 107.3087313
time: 10.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3087366, upper bound: 107.3087313
time: 18.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3092364, upper bound: 107.3092360
time: 10.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3092375, upper bound: 107.3092355
time: 10.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340945, upper bound: 107.3340923
time: 10.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340945, upper bound: 107.3340923
time: 10.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3307008, upper bound: 107.3307018
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3307008, upper bound: 107.3307018
time: 9.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2944137, upper bound: 107.2944219
time: 23.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2944137, upper bound: 107.2944222
time: 9.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3108682, upper bound: 107.3108666
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3108682, upper bound: 107.3108666
time: 10.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3295159, upper bound: 107.3295125
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3295159, upper bound: 107.3295125
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3172218, upper bound: 107.3172230
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3172218, upper bound: 107.3172230
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2837897, upper bound: 107.2837890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2837897, upper bound: 107.2837890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3231855, upper bound: 107.3231839
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3231860, upper bound: 107.3231835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2856335, upper bound: 107.2856328
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2856335, upper bound: 107.2856328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2968744, upper bound: 107.2968749
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2968743, upper bound: 107.2968753
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3087366, upper bound: 107.3087313
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3087366, upper bound: 107.3087313
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3092364, upper bound: 107.3092360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3092375, upper bound: 107.3092355
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3340945, upper bound: 107.3340923
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3340945, upper bound: 107.3340923
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3307008, upper bound: 107.3307018
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3307008, upper bound: 107.3307018
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2944137, upper bound: 107.2944219
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.2944137, upper bound: 107.2944222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3108682, upper bound: 107.3108666
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -107.3108682, upper bound: 107.3108666
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.23
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.23
Output dim: 7, lower bound: -107.3061016, upper bound: 107.3061067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.23
Output dim: 7, lower bound: -107.3105398, upper bound: 107.3105439
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.23
Output dim: 7, lower bound: -107.3105422, upper bound: 107.3105414
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3200897, upper bound: 107.3200897
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3200897, upper bound: 107.3200897
time: 7.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.66
Output dim: 7, lower bound: -107.3200897, upper bound: 107.3200897
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.66
Output dim: 7, lower bound: -107.3200897, upper bound: 107.3200897

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3189589, upper bound: 107.3189593
time: 10.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3189593, upper bound: 107.3189589
time: 10.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2925916, upper bound: 107.2925916
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2925916, upper bound: 107.2925916
time: 6.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 7, lower bound: -107.3189589, upper bound: 107.3189593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 7, lower bound: -107.3189593, upper bound: 107.3189589
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 7, lower bound: -107.2925916, upper bound: 107.2925916
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 7, lower bound: -107.2925916, upper bound: 107.2925916

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3185035, upper bound: 107.3185035
time: 10.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3185035, upper bound: 107.3185035
time: 10.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
time: 9.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907345
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907345
time: 20.92 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 35.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.3185035, upper bound: 107.3185035
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.3185035, upper bound: 107.3185035
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916070
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907345
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.70
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907345

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
time: 8.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
time: 8.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3184997, upper bound: 107.3184999
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3184999, upper bound: 107.3184997
time: 9.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
time: 11.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
time: 10.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
time: 11.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
time: 11.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
time: 10.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895883, upper bound: 107.2895881
time: 8.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907339, upper bound: 107.2907345
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907339
time: 6.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.3184997, upper bound: 107.3184999
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.3184999, upper bound: 107.3184997
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2895883, upper bound: 107.2895881
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2907339, upper bound: 107.2907345
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.55
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907339

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2849127, upper bound: 107.2849134
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2849127, upper bound: 107.2849134
time: 10.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895687, upper bound: 107.2895690
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895687, upper bound: 107.2895688
time: 8.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870261, upper bound: 107.2870269
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870261, upper bound: 107.2870269
time: 11.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895821, upper bound: 107.2895820
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895818, upper bound: 107.2895823
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3153387, upper bound: 107.3153387
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3153384, upper bound: 107.3153388
time: 12.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3184992, upper bound: 107.3184997
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3184999, upper bound: 107.3184991
time: 9.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2868078, upper bound: 107.2868079
time: 12.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2868082, upper bound: 107.2868078
time: 8.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915301
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
time: 9.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
time: 9.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414
1: -55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092
2: -70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666
3: -81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354
4: -72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363
5: -62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670
6: -60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196
7: -69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519
8: -77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454
9: -60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2846400, upper bound: 107.2846401
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2846400, upper bound: 107.2846400
time: 8.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2849127, upper bound: 107.2849134
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2849127, upper bound: 107.2849134
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2895687, upper bound: 107.2895690
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2895687, upper bound: 107.2895688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2870261, upper bound: 107.2870269
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2870261, upper bound: 107.2870269
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2895821, upper bound: 107.2895820
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2895818, upper bound: 107.2895823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.3153387, upper bound: 107.3153387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.3153384, upper bound: 107.3153388
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.3184992, upper bound: 107.3184997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.3184999, upper bound: 107.3184991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2868078, upper bound: 107.2868079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2868082, upper bound: 107.2868078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2915302, upper bound: 107.2915303
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2870530, upper bound: 107.2870530
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2846400, upper bound: 107.2846401
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 7, lower bound: -107.2846400, upper bound: 107.2846400
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2916069, upper bound: 107.2916069
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2895881, upper bound: 107.2895883
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2895883, upper bound: 107.2895881
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2907339, upper bound: 107.2907345
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 7, lower bound: -107.2907345, upper bound: 107.2907339
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1817.45 seconds
