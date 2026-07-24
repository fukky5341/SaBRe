## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 71.4340763181


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693)
1: (-36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435)
2: (-47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022)
3: (-53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733)
4: (-47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030)
5: (-42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557)
6: (-39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417)
7: (-45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556)
8: (-52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380)
9: (-39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.69 + 13.35 = 16.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051676, upper bound: 71.5051671
time: 8.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051671, upper bound: 71.5051676
time: 9.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.75
Output dim: 7, lower bound: -71.5051676, upper bound: 71.5051671
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.75
Output dim: 7, lower bound: -71.5051671, upper bound: 71.5051676

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5051057
time: 9.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5051057
time: 7.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5051069
time: 9.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5051069
time: 10.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 22.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5051057
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5051057
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5051069
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5051069

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5050873
time: 9.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5050874, upper bound: 71.5051057
time: 10.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5050873
time: 9.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5050874, upper bound: 71.5051057
time: 10.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5050873
time: 7.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5050873, upper bound: 71.5051069
time: 10.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5050873
time: 9.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5050873, upper bound: 71.5051069
time: 11.42 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 23.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5050873
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5050874, upper bound: 71.5051057
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5051069, upper bound: 71.5050873
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5050874, upper bound: 71.5051057
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5050873
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5050873, upper bound: 71.5051069
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5051057, upper bound: 71.5050873
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.10
Output dim: 7, lower bound: -71.5050873, upper bound: 71.5051069

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049045, upper bound: 71.5048839
time: 8.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048870
time: 9.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048868, upper bound: 71.5048977
time: 9.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049037
time: 8.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049045, upper bound: 71.5048839
time: 8.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048870
time: 9.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048868, upper bound: 71.5048977
time: 11.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049037
time: 10.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049037, upper bound: 71.5048839
time: 9.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048868
time: 7.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048870, upper bound: 71.5048977
time: 9.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049045
time: 9.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049037, upper bound: 71.5048839
time: 9.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5048868
time: 8.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048870, upper bound: 71.5048977
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049045
time: 9.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 19.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5049045, upper bound: 71.5048839
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048870
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048868, upper bound: 71.5048977
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049037
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5049045, upper bound: 71.5048839
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048870
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048868, upper bound: 71.5048977
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049037
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5049037, upper bound: 71.5048839
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048977, upper bound: 71.5048868
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048870, upper bound: 71.5048977
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049045
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5049037, upper bound: 71.5048839
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5048868
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048870, upper bound: 71.5048977
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.93
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049045

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946362, upper bound: 71.4946254
time: 8.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946362, upper bound: 71.4946254
time: 6.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946281
time: 8.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946345, upper bound: 71.4946281
time: 8.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946296
time: 8.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946295
time: 8.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946287, upper bound: 71.4946324
time: 8.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946287, upper bound: 71.4946324
time: 11.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946349, upper bound: 71.4946252
time: 7.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946349, upper bound: 71.4946252
time: 7.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946279
time: 21.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946279
time: 20.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946311
time: 9.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946311
time: 8.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946284, upper bound: 71.4946339
time: 9.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946284, upper bound: 71.4946339
time: 9.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946339, upper bound: 71.4946284
time: 7.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946339, upper bound: 71.4946284
time: 7.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946307
time: 10.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946311, upper bound: 71.4946307
time: 8.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946279, upper bound: 71.4946319
time: 9.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946279, upper bound: 71.4946319
time: 8.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946349
time: 9.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946349
time: 9.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693
1: -36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435
2: -47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022
3: -53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733
4: -47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030
5: -42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557
6: -39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417
7: -45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556
8: -52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380
9: -39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946324, upper bound: 71.4946287
time: 33.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4946324, upper bound: 71.4946287
time: 7.98 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 44.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946362, upper bound: 71.4946254
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946362, upper bound: 71.4946254
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946281
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946345, upper bound: 71.4946281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946296
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946295
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946287, upper bound: 71.4946324
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946287, upper bound: 71.4946324
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946349, upper bound: 71.4946252
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946349, upper bound: 71.4946252
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946279
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946319, upper bound: 71.4946279
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946311
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946307, upper bound: 71.4946311
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946284, upper bound: 71.4946339
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946284, upper bound: 71.4946339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946339, upper bound: 71.4946284
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946339, upper bound: 71.4946284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946307
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946311, upper bound: 71.4946307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946279, upper bound: 71.4946319
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946279, upper bound: 71.4946319
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946349
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946252, upper bound: 71.4946349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946324, upper bound: 71.4946287
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 44.26
Output dim: 7, lower bound: -71.4946324, upper bound: 71.4946287
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.26
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5048868
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.26
Output dim: 7, lower bound: -71.5048870, upper bound: 71.5048977
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.26
Output dim: 7, lower bound: -71.5048839, upper bound: 71.5049045

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 16.04 + 623.05 = 639.09 seconds
