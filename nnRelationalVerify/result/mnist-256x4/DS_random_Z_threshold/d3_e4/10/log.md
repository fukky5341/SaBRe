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
execution time: IAR + RelationalAnalysis = 0.92 + 13.39 = 14.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7150558, upper bound: 154.7150558

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7026202, upper bound: 154.7026212
time: 9.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7026212, upper bound: 154.7026202
time: 9.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.52 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.52
Output dim: 4, lower bound: -154.7026202, upper bound: 154.7026212
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.52
Output dim: 4, lower bound: -154.7026212, upper bound: 154.7026202

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6743180, upper bound: 154.6743180
time: 11.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6743180, upper bound: 154.6743180
time: 8.54 seconds

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
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6984106, upper bound: 154.6984106
time: 10.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6984106, upper bound: 154.6984106
time: 7.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.58 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.58
Output dim: 4, lower bound: -154.6743180, upper bound: 154.6743180
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.58
Output dim: 4, lower bound: -154.6743180, upper bound: 154.6743180
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.58
Output dim: 4, lower bound: -154.6984106, upper bound: 154.6984106
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.58
Output dim: 4, lower bound: -154.6984106, upper bound: 154.6984106

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6740739, upper bound: 154.6740742
time: 8.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6740742, upper bound: 154.6740739
time: 7.87 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6680424, upper bound: 154.6680414
time: 9.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6680414, upper bound: 154.6680424
time: 10.81 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874381, upper bound: 154.6874381
time: 9.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874381, upper bound: 154.6874381
time: 9.85 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6291447, upper bound: 154.6291447
time: 8.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6291447, upper bound: 154.6291447
time: 8.64 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 18.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6740739, upper bound: 154.6740742
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6740742, upper bound: 154.6740739
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6680424, upper bound: 154.6680414
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6680414, upper bound: 154.6680424
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6874381, upper bound: 154.6874381
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6874381, upper bound: 154.6874381
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6291447, upper bound: 154.6291447
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.27
Output dim: 4, lower bound: -154.6291447, upper bound: 154.6291447

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6330104, upper bound: 154.6330107
time: 10.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6330104, upper bound: 154.6330107
time: 9.31 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5563365, upper bound: 154.5563365
time: 8.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5563365, upper bound: 154.5563365
time: 9.60 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
time: 8.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
time: 8.37 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6628492, upper bound: 154.6628506
time: 13.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6628502, upper bound: 154.6628498
time: 8.58 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874377, upper bound: 154.6874373
time: 8.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874373, upper bound: 154.6874377
time: 9.58 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6852309, upper bound: 154.6852348
time: 9.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6852350, upper bound: 154.6852309
time: 9.15 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6227430, upper bound: 154.6227415
time: 9.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6227415, upper bound: 154.6227430
time: 8.57 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6282791, upper bound: 154.6282781
time: 8.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6282781, upper bound: 154.6282791
time: 14.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 25.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6330104, upper bound: 154.6330107
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6330104, upper bound: 154.6330107
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.5563365, upper bound: 154.5563365
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.5563365, upper bound: 154.5563365
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6628492, upper bound: 154.6628506
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6628502, upper bound: 154.6628498
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6874377, upper bound: 154.6874373
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6874373, upper bound: 154.6874377
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6852309, upper bound: 154.6852348
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6852350, upper bound: 154.6852309
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6227430, upper bound: 154.6227415
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6227415, upper bound: 154.6227430
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6282791, upper bound: 154.6282781
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 25.63
Output dim: 4, lower bound: -154.6282781, upper bound: 154.6282791

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6325564, upper bound: 154.6325561
time: 9.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6325564, upper bound: 154.6325561
time: 11.27 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
time: 9.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6257919, upper bound: 154.6257947
time: 7.04 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.4983774, upper bound: 154.4983704
time: 6.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.4983774, upper bound: 154.4983704
time: 8.94 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5719271, upper bound: 154.5719154
time: 8.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5719271, upper bound: 154.5719154
time: 8.99 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6448807, upper bound: 154.6448913
time: 9.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6448807, upper bound: 154.6448913
time: 8.54 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6567827, upper bound: 154.6567839
time: 11.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6567827, upper bound: 154.6567839
time: 12.40 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874299, upper bound: 154.6874298
time: 7.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6874299, upper bound: 154.6874298
time: 8.82 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6824424, upper bound: 154.6824288
time: 7.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6824411, upper bound: 154.6824282
time: 8.21 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5808721, upper bound: 154.5808737
time: 7.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5808721, upper bound: 154.5808737
time: 7.05 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6842995, upper bound: 154.6842865
time: 8.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6842995, upper bound: 154.6842865
time: 8.25 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6226357, upper bound: 154.6226348
time: 8.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6226357, upper bound: 154.6226348
time: 8.46 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5968605, upper bound: 154.5968616
time: 7.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5968605, upper bound: 154.5968616
time: 10.39 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6251644, upper bound: 154.6251612
time: 9.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6251649, upper bound: 154.6251592
time: 8.62 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6282670, upper bound: 154.6282791
time: 8.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6282781, upper bound: 154.6282701
time: 13.41 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6325564, upper bound: 154.6325561
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6325564, upper bound: 154.6325561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6257948, upper bound: 154.6257924
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6257919, upper bound: 154.6257947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.4983774, upper bound: 154.4983704
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.4983774, upper bound: 154.4983704
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5719271, upper bound: 154.5719154
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5719271, upper bound: 154.5719154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6448807, upper bound: 154.6448913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6448807, upper bound: 154.6448913
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6567827, upper bound: 154.6567839
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6567827, upper bound: 154.6567839
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6874299, upper bound: 154.6874298
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6874299, upper bound: 154.6874298
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6824424, upper bound: 154.6824288
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6824411, upper bound: 154.6824282
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5808721, upper bound: 154.5808737
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5808721, upper bound: 154.5808737
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6842995, upper bound: 154.6842865
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6842995, upper bound: 154.6842865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6226357, upper bound: 154.6226348
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6226357, upper bound: 154.6226348
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5968605, upper bound: 154.5968616
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.5968605, upper bound: 154.5968616
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6251644, upper bound: 154.6251612
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6251649, upper bound: 154.6251592
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6282670, upper bound: 154.6282791
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.70
Output dim: 4, lower bound: -154.6282781, upper bound: 154.6282701

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 14.31 + 597.75 = 612.06 seconds
