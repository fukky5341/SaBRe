## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 154.56034074419998
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.43 + 10.00 = 11.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7151429, upper bound: 154.7151428


# Binary Search by BASE starts (time budget: 1988.57 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search Result
Binary search time: 46.17 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1942.40 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141606, upper bound: 154.7141612
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141612, upper bound: 154.7141606
time: 11.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.38
Output dim: 4, lower bound: -154.7141606, upper bound: 154.7141612
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.38
Output dim: 4, lower bound: -154.7141612, upper bound: 154.7141606

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115205, upper bound: 154.7115217
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115205, upper bound: 154.7115217
time: 7.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115217, upper bound: 154.7115205
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115217, upper bound: 154.7115205
time: 8.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.48
Output dim: 4, lower bound: -154.7115205, upper bound: 154.7115217
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.48
Output dim: 4, lower bound: -154.7115205, upper bound: 154.7115217
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.48
Output dim: 4, lower bound: -154.7115217, upper bound: 154.7115205
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.48
Output dim: 4, lower bound: -154.7115217, upper bound: 154.7115205

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096274, upper bound: 154.7096279
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096278, upper bound: 154.7096279
time: 8.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096274, upper bound: 154.7096279
time: 9.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096278, upper bound: 154.7096279
time: 8.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096278
time: 11.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096274
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096278
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096274
time: 8.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096274, upper bound: 154.7096279
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096278, upper bound: 154.7096279
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096274, upper bound: 154.7096279
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096278, upper bound: 154.7096279
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096278
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096274
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096278
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.46
Output dim: 4, lower bound: -154.7096279, upper bound: 154.7096274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994955, upper bound: 154.6994921
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994957, upper bound: 154.6994918
time: 15.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994945, upper bound: 154.6994929
time: 11.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994949, upper bound: 154.6994928
time: 8.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994955, upper bound: 154.6994921
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994957, upper bound: 154.6994918
time: 9.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994945, upper bound: 154.6994929
time: 11.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994949, upper bound: 154.6994928
time: 9.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994941, upper bound: 154.6994936
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994942, upper bound: 154.6994932
time: 8.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994932, upper bound: 154.6994957
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994935, upper bound: 154.6994955
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994941, upper bound: 154.6994936
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994942, upper bound: 154.6994932
time: 7.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994932, upper bound: 154.6994957
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6994935, upper bound: 154.6994955
time: 6.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994955, upper bound: 154.6994921
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994957, upper bound: 154.6994918
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994945, upper bound: 154.6994929
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994949, upper bound: 154.6994928
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994955, upper bound: 154.6994921
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994957, upper bound: 154.6994918
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994945, upper bound: 154.6994929
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994949, upper bound: 154.6994928
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994941, upper bound: 154.6994936
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994942, upper bound: 154.6994932
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994932, upper bound: 154.6994957
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994935, upper bound: 154.6994955
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994941, upper bound: 154.6994936
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994942, upper bound: 154.6994932
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994932, upper bound: 154.6994957
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.77
Output dim: 4, lower bound: -154.6994935, upper bound: 154.6994955

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 6.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.33
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141667, upper bound: 154.7141684
time: 12.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141684, upper bound: 154.7141667
time: 7.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.97
Output dim: 4, lower bound: -154.7141667, upper bound: 154.7141684
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.97
Output dim: 4, lower bound: -154.7141684, upper bound: 154.7141667

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115312, upper bound: 154.7115322
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115312, upper bound: 154.7115322
time: 8.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115322, upper bound: 154.7115312
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115322, upper bound: 154.7115312
time: 8.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.46
Output dim: 4, lower bound: -154.7115312, upper bound: 154.7115322
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.46
Output dim: 4, lower bound: -154.7115312, upper bound: 154.7115322
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.46
Output dim: 4, lower bound: -154.7115322, upper bound: 154.7115312
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.46
Output dim: 4, lower bound: -154.7115322, upper bound: 154.7115312

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096407, upper bound: 154.7096415
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096408, upper bound: 154.7096416
time: 7.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096407, upper bound: 154.7096415
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096408, upper bound: 154.7096416
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096416, upper bound: 154.7096408
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096415, upper bound: 154.7096407
time: 8.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096416, upper bound: 154.7096408
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096415, upper bound: 154.7096407
time: 8.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096407, upper bound: 154.7096415
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096408, upper bound: 154.7096416
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096407, upper bound: 154.7096415
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096408, upper bound: 154.7096416
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096416, upper bound: 154.7096408
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096415, upper bound: 154.7096407
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096416, upper bound: 154.7096408
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.60
Output dim: 4, lower bound: -154.7096415, upper bound: 154.7096407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995146, upper bound: 154.6995111
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995155, upper bound: 154.6995106
time: 7.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995124, upper bound: 154.6995121
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995135, upper bound: 154.6995128
time: 9.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995146, upper bound: 154.6995098
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995155, upper bound: 154.6995106
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995124, upper bound: 154.6995121
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995135, upper bound: 154.6995128
time: 11.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995128, upper bound: 154.6995135
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995134, upper bound: 154.6995124
time: 8.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995106, upper bound: 154.6995142
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995111, upper bound: 154.6995133
time: 8.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995128, upper bound: 154.6995122
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995134, upper bound: 154.6995124
time: 9.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995106, upper bound: 154.6995142
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995111, upper bound: 154.6995133
time: 8.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995146, upper bound: 154.6995111
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995155, upper bound: 154.6995106
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995124, upper bound: 154.6995121
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995135, upper bound: 154.6995128
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995146, upper bound: 154.6995098
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995155, upper bound: 154.6995106
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995124, upper bound: 154.6995121
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995135, upper bound: 154.6995128
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995128, upper bound: 154.6995135
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995134, upper bound: 154.6995124
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995106, upper bound: 154.6995142
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995111, upper bound: 154.6995133
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995128, upper bound: 154.6995122
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995134, upper bound: 154.6995124
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995106, upper bound: 154.6995142
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.97
Output dim: 4, lower bound: -154.6995111, upper bound: 154.6995133

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.41
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=168.67294311523438
rel_dist={4: [-154.7151344301911, 154.71513443019103]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141698, upper bound: 154.7141719
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141719, upper bound: 154.7141698
time: 8.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.60
Output dim: 4, lower bound: -154.7141698, upper bound: 154.7141719
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.60
Output dim: 4, lower bound: -154.7141719, upper bound: 154.7141698

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115365, upper bound: 154.7115374
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115365, upper bound: 154.7115374
time: 6.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115374, upper bound: 154.7115365
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115374, upper bound: 154.7115365
time: 7.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 4, lower bound: -154.7115365, upper bound: 154.7115374
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 4, lower bound: -154.7115365, upper bound: 154.7115374
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 4, lower bound: -154.7115374, upper bound: 154.7115365
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 4, lower bound: -154.7115374, upper bound: 154.7115365

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096477, upper bound: 154.7096490
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096484, upper bound: 154.7096491
time: 8.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096477, upper bound: 154.7096490
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096484, upper bound: 154.7096491
time: 8.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096491, upper bound: 154.7096484
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096490, upper bound: 154.7096477
time: 7.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096491, upper bound: 154.7096484
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096490, upper bound: 154.7096477
time: 6.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096477, upper bound: 154.7096490
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096484, upper bound: 154.7096491
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096477, upper bound: 154.7096490
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096484, upper bound: 154.7096491
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096491, upper bound: 154.7096484
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096490, upper bound: 154.7096477
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096491, upper bound: 154.7096484
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 4, lower bound: -154.7096490, upper bound: 154.7096477

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995257, upper bound: 154.6995225
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995268, upper bound: 154.6995206
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995230, upper bound: 154.6995242
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995241, upper bound: 154.6995234
time: 8.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995257, upper bound: 154.6995226
time: 9.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995268, upper bound: 154.6995219
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995230, upper bound: 154.6995242
time: 7.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995241, upper bound: 154.6995234
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995234, upper bound: 154.6995241
time: 10.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995242, upper bound: 154.6995230
time: 8.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995206, upper bound: 154.6995281
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995212, upper bound: 154.6995270
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995247, upper bound: 154.6995254
time: 15.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995242, upper bound: 154.6995230
time: 7.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995219, upper bound: 154.6995268
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995212, upper bound: 154.6995257
time: 8.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995257, upper bound: 154.6995225
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995268, upper bound: 154.6995206
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995230, upper bound: 154.6995242
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995241, upper bound: 154.6995234
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995257, upper bound: 154.6995226
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995268, upper bound: 154.6995219
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995230, upper bound: 154.6995242
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995241, upper bound: 154.6995234
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995234, upper bound: 154.6995241
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995242, upper bound: 154.6995230
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995206, upper bound: 154.6995281
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995212, upper bound: 154.6995270
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995247, upper bound: 154.6995254
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995242, upper bound: 154.6995230
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995219, upper bound: 154.6995268
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 4, lower bound: -154.6995212, upper bound: 154.6995257

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.19
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=168.67294311523438
rel_dist={4: [-154.7151401959666, 154.71514022107215]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141712, upper bound: 154.7141735
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7141735, upper bound: 154.7141712
time: 5.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.04
Output dim: 4, lower bound: -154.7141712, upper bound: 154.7141735
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.04
Output dim: 4, lower bound: -154.7141735, upper bound: 154.7141712

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115389, upper bound: 154.7115396
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115389, upper bound: 154.7115396
time: 6.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115396, upper bound: 154.7115389
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7115396, upper bound: 154.7115389
time: 7.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 4, lower bound: -154.7115389, upper bound: 154.7115396
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 4, lower bound: -154.7115389, upper bound: 154.7115396
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 4, lower bound: -154.7115396, upper bound: 154.7115389
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 4, lower bound: -154.7115396, upper bound: 154.7115389

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096507, upper bound: 154.7096527
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096517, upper bound: 154.7096527
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096507, upper bound: 154.7096527
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096517, upper bound: 154.7096527
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096517
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096507
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096517
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096507
time: 9.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096507, upper bound: 154.7096527
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096517, upper bound: 154.7096527
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096507, upper bound: 154.7096527
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096517, upper bound: 154.7096527
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096517
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096507
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096517
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 4, lower bound: -154.7096527, upper bound: 154.7096507

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995319, upper bound: 154.6995283
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995331, upper bound: 154.6995276
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995302
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995301, upper bound: 154.6995293
time: 7.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995319, upper bound: 154.6995283
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995331, upper bound: 154.6995263
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995289, upper bound: 154.6995302
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995301, upper bound: 154.6995293
time: 6.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995293, upper bound: 154.6995314
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995289
time: 10.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995263, upper bound: 154.6995331
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995270, upper bound: 154.6995332
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995293, upper bound: 154.6995314
time: 10.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995302
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995263, upper bound: 154.6995331
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6995270, upper bound: 154.6995319
time: 6.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995319, upper bound: 154.6995283
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995331, upper bound: 154.6995276
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995302
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995301, upper bound: 154.6995293
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995319, upper bound: 154.6995283
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995331, upper bound: 154.6995263
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995289, upper bound: 154.6995302
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995301, upper bound: 154.6995293
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995293, upper bound: 154.6995314
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995289
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995263, upper bound: 154.6995331
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995270, upper bound: 154.6995332
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995293, upper bound: 154.6995314
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995302, upper bound: 154.6995302
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995263, upper bound: 154.6995331
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 4, lower bound: -154.6995270, upper bound: 154.6995319

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
time: 5.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.89
Output dim: 4, lower bound: -154.5567354, upper bound: 154.5567348
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.89
Output dim: 4, lower bound: -154.6995263, upper bound: 154.6995331
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.89
Output dim: 4, lower bound: -154.6995270, upper bound: 154.6995319
Binary search (step 3): status=Status.UNKNOWN, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=168.67294311523438
rel_dist={4: [-154.7151428770338, 154.71514282635343]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.04296875
execution time: 1942.83 seconds
