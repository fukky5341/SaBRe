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
execution time: IAR + RelationalAnalysis = 0.89 + 13.15 = 14.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3454662, upper bound: 107.3454661

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422903
time: 8.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422903
time: 8.22 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 16.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 16.26
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422903
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 16.26
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422903

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
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422900
time: 8.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422900, upper bound: 107.3422903
time: 9.82 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096621
time: 9.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096621
time: 9.61 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 20.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.05
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422900
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.05
Output dim: 7, lower bound: -107.3422900, upper bound: 107.3422903
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.05
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096621
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.05
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096621

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422897
time: 11.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3422899, upper bound: 107.3422900
time: 9.45 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
time: 8.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
time: 9.02 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3087559, upper bound: 107.3087541
time: 8.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3087541, upper bound: 107.3087552
time: 8.72 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070643
time: 9.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070643
time: 8.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 17.93 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3422904, upper bound: 107.3422897
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3422899, upper bound: 107.3422900
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3087559, upper bound: 107.3087541
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3087541, upper bound: 107.3087552
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070643
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070643

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959653
time: 9.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959653
time: 9.54 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963073
time: 9.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963073
time: 9.93 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070615
time: 9.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070642, upper bound: 107.3070615
time: 7.95 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
time: 8.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096585
time: 8.52 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2990251, upper bound: 107.2990242
time: 10.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2990251, upper bound: 107.2990242
time: 8.45 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993282
time: 8.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993282
time: 8.96 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070642, upper bound: 107.3070630
time: 8.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070630, upper bound: 107.3070642
time: 10.46 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3068820, upper bound: 107.3068811
time: 9.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3068811, upper bound: 107.3068820
time: 9.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 19.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959653
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959653
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963073
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963073
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070615
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3070642, upper bound: 107.3070615
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096586
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096585
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2990251, upper bound: 107.2990242
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2990251, upper bound: 107.2990242
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993282
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993282
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3070642, upper bound: 107.3070630
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3070630, upper bound: 107.3070642
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3068820, upper bound: 107.3068811
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.88
Output dim: 7, lower bound: -107.3068811, upper bound: 107.3068820

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959658
time: 7.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2959639, upper bound: 107.2959653
time: 7.46 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2750935, upper bound: 107.2750980
time: 7.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2750935, upper bound: 107.2750980
time: 7.78 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2873830, upper bound: 107.2873876
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2873830, upper bound: 107.2873876
time: 9.00 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2963040, upper bound: 107.2963073
time: 6.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963051
time: 7.05 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970819
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970819
time: 7.18 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070640, upper bound: 107.3070615
time: 9.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070615
time: 9.60 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096594, upper bound: 107.3096586
time: 7.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096533
time: 10.89 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096582
time: 11.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096585
time: 9.65 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968070, upper bound: 107.2968063
time: 8.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2968063, upper bound: 107.2968064
time: 7.32 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2754122, upper bound: 107.2754081
time: 7.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2754122, upper bound: 107.2754081
time: 7.89 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974574, upper bound: 107.2974575
time: 9.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2974574, upper bound: 107.2974575
time: 6.87 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2993279, upper bound: 107.2993282
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993281
time: 6.31 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970863
time: 11.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970863
time: 7.86 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070630, upper bound: 107.3070623
time: 9.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3070604, upper bound: 107.3070643
time: 7.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2879169, upper bound: 107.2879168
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2879170, upper bound: 107.2879168
time: 8.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3040165, upper bound: 107.3040216
time: 9.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3040163, upper bound: 107.3040220
time: 8.97 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 23.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2959641, upper bound: 107.2959658
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2959639, upper bound: 107.2959653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2750935, upper bound: 107.2750980
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2750935, upper bound: 107.2750980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2873830, upper bound: 107.2873876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2873830, upper bound: 107.2873876
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2963040, upper bound: 107.2963073
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2963053, upper bound: 107.2963051
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970819
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970819
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3070640, upper bound: 107.3070615
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3070643, upper bound: 107.3070615
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3096594, upper bound: 107.3096586
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096533
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096582
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3096621, upper bound: 107.3096585
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2968070, upper bound: 107.2968063
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2968063, upper bound: 107.2968064
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2754122, upper bound: 107.2754081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2754122, upper bound: 107.2754081
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2974574, upper bound: 107.2974575
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2974574, upper bound: 107.2974575
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2993279, upper bound: 107.2993282
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2993281, upper bound: 107.2993281
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2970863, upper bound: 107.2970863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3070630, upper bound: 107.3070623
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3070604, upper bound: 107.3070643
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2879169, upper bound: 107.2879168
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.2879170, upper bound: 107.2879168
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3040165, upper bound: 107.3040216
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 7, lower bound: -107.3040163, upper bound: 107.3040220

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 14.04 + 601.09 = 615.13 seconds
