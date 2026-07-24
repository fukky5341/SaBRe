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
execution time: IAR + RelationalAnalysis = 0.85 + 12.60 = 13.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025247
time: 9.82 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025247
time: 10.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 19.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 19.94
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025247
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 19.94
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025247

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025204
time: 8.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5025204, upper bound: 71.5025247
time: 11.13 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4662373, upper bound: 71.4662373
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4662373, upper bound: 71.4662373
time: 7.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.92
Output dim: 7, lower bound: -71.5025247, upper bound: 71.5025204
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.92
Output dim: 7, lower bound: -71.5025204, upper bound: 71.5025247
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.92
Output dim: 7, lower bound: -71.4662373, upper bound: 71.4662373
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.92
Output dim: 7, lower bound: -71.4662373, upper bound: 71.4662373

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4902637, upper bound: 71.4902636
time: 6.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4902637, upper bound: 71.4902636
time: 6.62 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4919870, upper bound: 71.4919876
time: 8.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4919870, upper bound: 71.4919876
time: 8.98 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541729
time: 6.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541729
time: 7.66 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4647420, upper bound: 71.4647231
time: 8.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4647231, upper bound: 71.4647420
time: 7.56 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 17.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4902637, upper bound: 71.4902636
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4902637, upper bound: 71.4902636
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4919870, upper bound: 71.4919876
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4919870, upper bound: 71.4919876
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541729
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541729
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4647420, upper bound: 71.4647231
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.02
Output dim: 7, lower bound: -71.4647231, upper bound: 71.4647420

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876360
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876360
time: 9.43 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4652529, upper bound: 71.4652466
time: 7.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4652529, upper bound: 71.4652466
time: 7.33 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4914396, upper bound: 71.4914414
time: 9.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4914396, upper bound: 71.4914412
time: 9.12 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564498
time: 7.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564498
time: 7.63 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4504687, upper bound: 71.4504741
time: 8.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4504741, upper bound: 71.4504687
time: 7.27 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541726
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541726, upper bound: 71.4541729
time: 6.78 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4647420, upper bound: 71.4647144
time: 7.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4647159, upper bound: 71.4647231
time: 6.55 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586560
time: 6.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586560
time: 7.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 21.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876360
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876360
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4652529, upper bound: 71.4652466
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4652529, upper bound: 71.4652466
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4914396, upper bound: 71.4914414
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4914396, upper bound: 71.4914412
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564498
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564498
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4504687, upper bound: 71.4504741
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4504741, upper bound: 71.4504687
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4541729, upper bound: 71.4541726
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4541726, upper bound: 71.4541729
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4647420, upper bound: 71.4647144
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4647159, upper bound: 71.4647231
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586560
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586560

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876282
time: 8.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876349, upper bound: 71.4876360
time: 9.28 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876423, upper bound: 71.4876344
time: 8.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4876422, upper bound: 71.4876339
time: 7.89 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4646270, upper bound: 71.4646239
time: 7.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4646271, upper bound: 71.4646237
time: 7.16 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
time: 6.22 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
time: 9.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
time: 8.85 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
time: 8.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
time: 8.12 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564486, upper bound: 71.4564498
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564494
time: 6.97 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564481, upper bound: 71.4564498
time: 5.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564492
time: 8.27 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
time: 8.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
time: 8.40 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 253

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4498492, upper bound: 71.4498462
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4498507, upper bound: 71.4498459
time: 7.72 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4505493, upper bound: 71.4505471
time: 8.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4505483, upper bound: 71.4505487
time: 7.35 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541721, upper bound: 71.4541729
time: 6.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4541726, upper bound: 71.4541725
time: 7.88 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
time: 8.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
time: 7.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
time: 7.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
time: 8.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
time: 6.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
time: 8.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586376
time: 8.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4586374, upper bound: 71.4586560
time: 14.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 27.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4876432, upper bound: 71.4876282
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4876349, upper bound: 71.4876360
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4876423, upper bound: 71.4876344
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4876422, upper bound: 71.4876339
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4646270, upper bound: 71.4646239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4646271, upper bound: 71.4646237
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4564486, upper bound: 71.4564498
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564494
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4564481, upper bound: 71.4564498
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4498492, upper bound: 71.4498462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4498507, upper bound: 71.4498459
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4505493, upper bound: 71.4505471
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4505483, upper bound: 71.4505487
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4541721, upper bound: 71.4541729
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4541726, upper bound: 71.4541725
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586376
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.59
Output dim: 7, lower bound: -71.4586374, upper bound: 71.4586560

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4872709, upper bound: 71.4872544
time: 9.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4872695, upper bound: 71.4872566
time: 10.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4852672, upper bound: 71.4852739
time: 8.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4852672, upper bound: 71.4852728
time: 6.83 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.85
Output dim: 7, lower bound: -71.4872709, upper bound: 71.4872544
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.85
Output dim: 7, lower bound: -71.4872695, upper bound: 71.4872566
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.85
Output dim: 7, lower bound: -71.4852672, upper bound: 71.4852739
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.85
Output dim: 7, lower bound: -71.4852672, upper bound: 71.4852728
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4876423, upper bound: 71.4876344
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4876422, upper bound: 71.4876339
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4646270, upper bound: 71.4646239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4646271, upper bound: 71.4646237
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4562558, upper bound: 71.4562557
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4830296, upper bound: 71.4830303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4819564, upper bound: 71.4819576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4564486, upper bound: 71.4564498
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564494
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4564481, upper bound: 71.4564498
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4564488, upper bound: 71.4564492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4433753, upper bound: 71.4433788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4498492, upper bound: 71.4498462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4498507, upper bound: 71.4498459
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4505493, upper bound: 71.4505471
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4505483, upper bound: 71.4505487
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4541721, upper bound: 71.4541729
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4541726, upper bound: 71.4541725
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4600711, upper bound: 71.4600615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4361198, upper bound: 71.4361198
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4440080, upper bound: 71.4440078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4586426, upper bound: 71.4586376
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.85
Output dim: 7, lower bound: -71.4586374, upper bound: 71.4586560

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 13.45 + 590.62 = 604.06 seconds
