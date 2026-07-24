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
execution time: IAR + LP analysis = 1.36 + 11.86 = 13.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3456123, upper bound: 107.3456123


# Binary Search by BASE starts (time budget: 1986.78 seconds, max iter: 100)

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
Binary search time: 43.93 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1942.84 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864
time: 7.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.01
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.01
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448680, upper bound: 107.3448648
time: 10.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448643, upper bound: 107.3448685
time: 7.42 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448685, upper bound: 107.3448643
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448649, upper bound: 107.3448680
time: 9.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 7, lower bound: -107.3448680, upper bound: 107.3448648
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 7, lower bound: -107.3448643, upper bound: 107.3448685
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 7, lower bound: -107.3448685, upper bound: 107.3448643
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 7, lower bound: -107.3448649, upper bound: 107.3448680

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424528, upper bound: 107.3424434
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424486, upper bound: 107.3424454
time: 10.82 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424458, upper bound: 107.3424480
time: 10.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424443, upper bound: 107.3424516
time: 11.65 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424516, upper bound: 107.3424443
time: 9.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424480, upper bound: 107.3424457
time: 8.91 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424454, upper bound: 107.3424486
time: 10.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3424434, upper bound: 107.3424528
time: 8.99 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424528, upper bound: 107.3424434
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424486, upper bound: 107.3424454
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424458, upper bound: 107.3424480
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424443, upper bound: 107.3424516
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424516, upper bound: 107.3424443
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424480, upper bound: 107.3424457
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424454, upper bound: 107.3424486
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 7, lower bound: -107.3424434, upper bound: 107.3424528

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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385403, upper bound: 107.3385354
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385403, upper bound: 107.3385354
time: 8.78 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385396
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385396
time: 9.79 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385398, upper bound: 107.3385355
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385398, upper bound: 107.3385355
time: 9.61 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385400
time: 9.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385400
time: 9.38 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385400, upper bound: 107.3385354
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385400, upper bound: 107.3385354
time: 8.07 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385397
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385398
time: 11.74 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385396, upper bound: 107.3385355
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385396, upper bound: 107.3385355
time: 9.59 seconds

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

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385403
time: 10.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385403
time: 11.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385403, upper bound: 107.3385354
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385403, upper bound: 107.3385354
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385396
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385396
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385398, upper bound: 107.3385355
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385398, upper bound: 107.3385355
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385400
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385400
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385400, upper bound: 107.3385354
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385400, upper bound: 107.3385354
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385397
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385355, upper bound: 107.3385398
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385396, upper bound: 107.3385355
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385396, upper bound: 107.3385355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385403
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385403

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
time: 10.07 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345279, upper bound: 107.3345299
time: 15.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
time: 9.34 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345262, upper bound: 107.3345300
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345259, upper bound: 107.3345299
time: 7.89 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345262, upper bound: 107.3345300
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345259, upper bound: 107.3345299
time: 7.53 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345265, upper bound: 107.3345305
time: 10.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345266, upper bound: 107.3345306
time: 9.73 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345305
time: 10.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345266, upper bound: 107.3345306
time: 9.45 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345311
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345246, upper bound: 107.3345311
time: 8.49 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345311
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345246, upper bound: 107.3345311
time: 7.90 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345246
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345248
time: 6.60 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345246
time: 10.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345248
time: 8.36 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345306, upper bound: 107.3345266
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345305, upper bound: 107.3345265
time: 9.17 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345306, upper bound: 107.3345266
time: 10.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345305, upper bound: 107.3345265
time: 9.94 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345259
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345300, upper bound: 107.3345262
time: 8.88 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345259
time: 11.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345300, upper bound: 107.3345262
time: 9.33 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345278
time: 10.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345279
time: 10.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345279, upper bound: 107.3345299
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345278, upper bound: 107.3345299
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345262, upper bound: 107.3345300
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345259, upper bound: 107.3345299
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345262, upper bound: 107.3345300
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345259, upper bound: 107.3345299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345265, upper bound: 107.3345305
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345266, upper bound: 107.3345306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345305
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345266, upper bound: 107.3345306
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345311
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345246, upper bound: 107.3345311
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345248, upper bound: 107.3345311
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345246, upper bound: 107.3345311
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345248
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345311, upper bound: 107.3345248
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345306, upper bound: 107.3345266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345305, upper bound: 107.3345265
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345306, upper bound: 107.3345266
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345305, upper bound: 107.3345265
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345259
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345300, upper bound: 107.3345262
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345259
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345300, upper bound: 107.3345262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345278
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 7, lower bound: -107.3345299, upper bound: 107.3345279
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.44
Output dim: 7, lower bound: -107.3385354, upper bound: 107.3385403
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448926, upper bound: 107.3448924
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448924, upper bound: 107.3448926
time: 9.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.25
Output dim: 7, lower bound: -107.3448926, upper bound: 107.3448924
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.25
Output dim: 7, lower bound: -107.3448924, upper bound: 107.3448926

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447911, upper bound: 107.3447880
time: 11.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447881, upper bound: 107.3447909
time: 10.14 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447909, upper bound: 107.3447881
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447880, upper bound: 107.3447910
time: 9.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.14
Output dim: 7, lower bound: -107.3447911, upper bound: 107.3447880
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.14
Output dim: 7, lower bound: -107.3447881, upper bound: 107.3447909
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.14
Output dim: 7, lower bound: -107.3447909, upper bound: 107.3447881
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.14
Output dim: 7, lower bound: -107.3447880, upper bound: 107.3447910

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423936, upper bound: 107.3423891
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423920, upper bound: 107.3423904
time: 9.93 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423912, upper bound: 107.3423919
time: 11.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423895, upper bound: 107.3423928
time: 10.42 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423928, upper bound: 107.3423895
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423919, upper bound: 107.3423912
time: 8.71 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423904, upper bound: 107.3423920
time: 10.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423891, upper bound: 107.3423936
time: 9.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423936, upper bound: 107.3423891
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423920, upper bound: 107.3423904
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423912, upper bound: 107.3423919
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423895, upper bound: 107.3423928
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423928, upper bound: 107.3423895
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423919, upper bound: 107.3423912
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423904, upper bound: 107.3423920
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.46
Output dim: 7, lower bound: -107.3423891, upper bound: 107.3423936

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
time: 11.35 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385235
time: 10.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 10.08 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 10.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 10.44 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
time: 10.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
time: 10.71 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
time: 9.83 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 11.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 8.17 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 11.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385217
time: 10.41 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239
time: 11.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385235
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385217
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.51
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345056
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
time: 10.95 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345056
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
time: 11.48 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345064
time: 11.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345036, upper bound: 107.3345056
time: 11.28 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345057
time: 10.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345063
time: 11.85 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345056
time: 10.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345056
time: 12.27 seconds

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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345041, upper bound: 107.3345056
time: 14.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345063
time: 12.11 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345065
time: 18.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
time: 9.06 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345057
time: 9.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
time: 11.55 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345025
time: 10.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
time: 9.62 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345064, upper bound: 107.3345034
time: 10.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
time: 10.62 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
time: 11.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
time: 8.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345056
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345056
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345064
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345036, upper bound: 107.3345056
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345057
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345063
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345056
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345056
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345041, upper bound: 107.3345056
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345063
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345065
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345057
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345064, upper bound: 107.3345034
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.60
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385217
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447995, upper bound: 107.3447995
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447995, upper bound: 107.3447995
time: 10.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.03
Output dim: 7, lower bound: -107.3447995, upper bound: 107.3447995
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.03
Output dim: 7, lower bound: -107.3447995, upper bound: 107.3447995

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447001, upper bound: 107.3446990
time: 11.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3446990, upper bound: 107.3447000
time: 11.92 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447001, upper bound: 107.3446990
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3446990, upper bound: 107.3447000
time: 11.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.32
Output dim: 7, lower bound: -107.3447001, upper bound: 107.3446990
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.32
Output dim: 7, lower bound: -107.3446990, upper bound: 107.3447000
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.32
Output dim: 7, lower bound: -107.3447001, upper bound: 107.3446990
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.32
Output dim: 7, lower bound: -107.3446990, upper bound: 107.3447000

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422982, upper bound: 107.3422963
time: 11.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422974, upper bound: 107.3422967
time: 10.70 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422971, upper bound: 107.3422972
time: 10.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422965, upper bound: 107.3422977
time: 11.08 seconds

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

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422977, upper bound: 107.3422965
time: 12.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422972, upper bound: 107.3422971
time: 10.15 seconds

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

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422967, upper bound: 107.3422974
time: 10.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422963, upper bound: 107.3422982
time: 9.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422982, upper bound: 107.3422963
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422974, upper bound: 107.3422967
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422971, upper bound: 107.3422972
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422965, upper bound: 107.3422977
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422977, upper bound: 107.3422965
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422972, upper bound: 107.3422971
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422967, upper bound: 107.3422974
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 7, lower bound: -107.3422963, upper bound: 107.3422982

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
time: 10.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
time: 10.88 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385052
time: 11.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385052
time: 10.99 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
time: 12.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
time: 11.26 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385052
time: 10.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385052
time: 11.87 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
time: 13.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
time: 13.06 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
time: 12.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
time: 10.72 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
time: 8.76 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054
time: 10.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054
time: 18.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 30.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385052
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385052
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385052
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385052
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385048
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
time: 11.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
time: 9.69 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
time: 11.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
time: 16.21 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
time: 9.38 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
time: 11.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344180, upper bound: 107.3344193
time: 10.90 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344189
time: 11.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
time: 10.66 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344193
time: 11.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
time: 12.14 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
time: 13.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
time: 11.70 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344179, upper bound: 107.3344193
time: 11.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
time: 10.18 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344179
time: 11.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344184
time: 10.83 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344183
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344179
time: 11.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344189
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344180, upper bound: 107.3344193
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344189
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344193
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344193
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344179, upper bound: 107.3344193
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344183, upper bound: 107.3344194
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344179
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344184, upper bound: 107.3344184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 7, lower bound: -107.3344193, upper bound: 107.3344179
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385049, upper bound: 107.3385053
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385049
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -107.3385048, upper bound: 107.3385054
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1830.49 seconds
