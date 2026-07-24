## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 154.56034074419998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957)
1: (-70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644)
2: (-94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537)
3: (-99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775)
4: (-103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431)
5: (-81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278)
6: (-83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274)
7: (-88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160)
8: (-104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331)
9: (-84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.82 + 13.50 = 14.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7150558, upper bound: 154.7150558

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7140918, upper bound: 154.7140926
time: 8.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7140926, upper bound: 154.7140918
time: 9.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.31
Output dim: 4, lower bound: -154.7140918, upper bound: 154.7140926
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.31
Output dim: 4, lower bound: -154.7140926, upper bound: 154.7140918

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7114936, upper bound: 154.7114943
time: 8.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7114936, upper bound: 154.7114943
time: 10.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7114943, upper bound: 154.7114936
time: 8.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7114943, upper bound: 154.7114936
time: 9.37 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 4, lower bound: -154.7114936, upper bound: 154.7114943
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 4, lower bound: -154.7114936, upper bound: 154.7114943
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 4, lower bound: -154.7114943, upper bound: 154.7114936
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.78
Output dim: 4, lower bound: -154.7114943, upper bound: 154.7114936

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096012
time: 9.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096011, upper bound: 154.7095991
time: 7.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096012
time: 9.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096011, upper bound: 154.7095991
time: 7.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096011
time: 12.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096012, upper bound: 154.7095991
time: 8.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096011
time: 11.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096012, upper bound: 154.7095991
time: 14.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.98 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096012
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7096011, upper bound: 154.7095991
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096012
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7096011, upper bound: 154.7095991
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096011
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7096012, upper bound: 154.7095991
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7095991, upper bound: 154.7096011
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 4, lower bound: -154.7096012, upper bound: 154.7095991

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994726, upper bound: 154.6994705
time: 17.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994718, upper bound: 154.6994705
time: 11.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994723, upper bound: 154.6994701
time: 13.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994724, upper bound: 154.6994701
time: 11.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994726, upper bound: 154.6994705
time: 14.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994731, upper bound: 154.6994691
time: 10.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994723, upper bound: 154.6994701
time: 16.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994724, upper bound: 154.6994701
time: 14.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994711
time: 8.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994723
time: 7.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994691, upper bound: 154.6994718
time: 9.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994692, upper bound: 154.6994713
time: 11.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994711
time: 9.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994710
time: 10.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994705, upper bound: 154.6994718
time: 9.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994692, upper bound: 154.6994713
time: 10.43 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 20.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994726, upper bound: 154.6994705
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994718, upper bound: 154.6994705
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994723, upper bound: 154.6994701
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994724, upper bound: 154.6994701
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994726, upper bound: 154.6994705
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994731, upper bound: 154.6994691
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994723, upper bound: 154.6994701
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994724, upper bound: 154.6994701
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994711
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994723
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994691, upper bound: 154.6994718
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994692, upper bound: 154.6994713
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994711
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994715, upper bound: 154.6994710
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994705, upper bound: 154.6994718
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.24
Output dim: 4, lower bound: -154.6994692, upper bound: 154.6994713

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 9.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 9.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 7.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 8.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957
1: -70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644
2: -94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537
3: -99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775
4: -103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431
5: -81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278
6: -83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274
7: -88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160
8: -104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331
9: -84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.65 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 14.32 + 580.47 = 594.79 seconds
