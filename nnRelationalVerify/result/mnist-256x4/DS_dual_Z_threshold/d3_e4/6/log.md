## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 107.2381207338


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

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 13.44 = 14.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3454662, upper bound: 107.3454661

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448926, upper bound: 107.3448924
time: 8.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448924, upper bound: 107.3448926
time: 8.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 17.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 17.95
Output dim: 7, lower bound: -107.3448926, upper bound: 107.3448924
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 17.95
Output dim: 7, lower bound: -107.3448924, upper bound: 107.3448926

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447911, upper bound: 107.3447880
time: 11.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447881, upper bound: 107.3447909
time: 9.66 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447909, upper bound: 107.3447881
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447880, upper bound: 107.3447910
time: 8.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.93
Output dim: 7, lower bound: -107.3447911, upper bound: 107.3447880
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.93
Output dim: 7, lower bound: -107.3447881, upper bound: 107.3447909
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.93
Output dim: 7, lower bound: -107.3447909, upper bound: 107.3447881
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.93
Output dim: 7, lower bound: -107.3447880, upper bound: 107.3447910

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423936, upper bound: 107.3423891
time: 8.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423920, upper bound: 107.3423904
time: 9.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423912, upper bound: 107.3423919
time: 11.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423895, upper bound: 107.3423928
time: 9.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423928, upper bound: 107.3423895
time: 9.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423919, upper bound: 107.3423912
time: 8.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423904, upper bound: 107.3423920
time: 9.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3423891, upper bound: 107.3423936
time: 9.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 19.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423936, upper bound: 107.3423891
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423920, upper bound: 107.3423904
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423912, upper bound: 107.3423919
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423895, upper bound: 107.3423928
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423928, upper bound: 107.3423895
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423919, upper bound: 107.3423912
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423904, upper bound: 107.3423920
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.76
Output dim: 7, lower bound: -107.3423891, upper bound: 107.3423936

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
time: 9.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
time: 10.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385235
time: 10.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 9.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 9.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 10.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
time: 10.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
time: 10.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
time: 8.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
time: 9.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 10.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
time: 7.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
time: 10.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385217
time: 9.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
time: 8.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239
time: 10.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 20.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385239, upper bound: 107.3385217
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385235
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385237
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385216
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385236
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385236, upper bound: 107.3385217
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385217
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.03
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345056
time: 8.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
time: 10.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345056
time: 7.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
time: 11.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345064
time: 11.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345036, upper bound: 107.3345056
time: 11.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345057
time: 10.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345063
time: 11.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345056
time: 10.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345056
time: 11.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345041, upper bound: 107.3345056
time: 13.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345063
time: 10.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345065
time: 16.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
time: 8.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345057
time: 8.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
time: 10.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345025
time: 9.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
time: 8.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345064, upper bound: 107.3345034
time: 10.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
time: 10.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
time: 10.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
time: 8.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
time: 8.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
time: 10.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345027
time: 8.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345037
time: 10.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345036
time: 11.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345029
time: 7.86 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 20.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345056
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345056
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345043, upper bound: 107.3345055
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345064
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345036, upper bound: 107.3345056
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345037, upper bound: 107.3345057
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345063
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345056
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345056
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345041, upper bound: 107.3345056
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345040, upper bound: 107.3345063
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345065
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345035, upper bound: 107.3345057
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345034, upper bound: 107.3345057
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345045, upper bound: 107.3345025
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345064, upper bound: 107.3345034
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345026
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345034
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345056, upper bound: 107.3345027
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345037
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345063, upper bound: 107.3345036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.31
Output dim: 7, lower bound: -107.3345057, upper bound: 107.3345029
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.31
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385238
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.31
Output dim: 7, lower bound: -107.3385217, upper bound: 107.3385239

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 14.34 + 595.71 = 610.04 seconds
