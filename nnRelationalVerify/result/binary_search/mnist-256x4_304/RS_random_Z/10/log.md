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
execution time: IAR + LP analysis = 1.45 + 9.91 = 11.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7151429, upper bound: 154.7151428


# Binary Search by BASE starts (time budget: 1988.64 seconds, max iter: 100)

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
Binary search time: 45.65 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1942.99 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7151251, upper bound: 154.7151159
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7151159, upper bound: 154.7151251
time: 6.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.47
Output dim: 4, lower bound: -154.7151251, upper bound: 154.7151159
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.47
Output dim: 4, lower bound: -154.7151159, upper bound: 154.7151251

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7076262, upper bound: 154.7076088
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7076262, upper bound: 154.7076064
time: 6.93 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7064264, upper bound: 154.7064424
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7064264, upper bound: 154.7064424
time: 8.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.68
Output dim: 4, lower bound: -154.7076262, upper bound: 154.7076088
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.68
Output dim: 4, lower bound: -154.7076262, upper bound: 154.7076064
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.68
Output dim: 4, lower bound: -154.7064264, upper bound: 154.7064424
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.68
Output dim: 4, lower bound: -154.7064264, upper bound: 154.7064424

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7067254, upper bound: 154.7066960
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7067254, upper bound: 154.7066960
time: 7.27 seconds

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7042904, upper bound: 154.7042854
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7042904, upper bound: 154.7042854
time: 8.07 seconds

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7056815, upper bound: 154.7056881
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7056786, upper bound: 154.7057001
time: 8.27 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6466152, upper bound: 154.6466470
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6466152, upper bound: 154.6466470
time: 7.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7067254, upper bound: 154.7066960
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7067254, upper bound: 154.7066960
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7042904, upper bound: 154.7042854
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7042904, upper bound: 154.7042854
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7056815, upper bound: 154.7056881
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.7056786, upper bound: 154.7057001
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.6466152, upper bound: 154.6466470
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 4, lower bound: -154.6466152, upper bound: 154.6466470

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607420
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607420
time: 9.28 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7033195, upper bound: 154.7032834
time: 10.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7033195, upper bound: 154.7032834
time: 6.54 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
time: 7.29 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
time: 7.31 seconds

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7029375, upper bound: 154.7029733
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7029497, upper bound: 154.7029511
time: 7.58 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6408072, upper bound: 154.6408717
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6408072, upper bound: 154.6408717
time: 6.52 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6463650, upper bound: 154.6463968
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6463662, upper bound: 154.6463976
time: 7.12 seconds

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
time: 8.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607420
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607420
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.7033195, upper bound: 154.7032834
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.7033195, upper bound: 154.7032834
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.7029375, upper bound: 154.7029733
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.7029497, upper bound: 154.7029511
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6408072, upper bound: 154.6408717
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6408072, upper bound: 154.6408717
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6463650, upper bound: 154.6463968
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.6463662, upper bound: 154.6463976
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.08
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6175353, upper bound: 154.6175276
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6175353, upper bound: 154.6175276
time: 9.47 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607411
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6607540, upper bound: 154.6607420
time: 7.99 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6927344, upper bound: 154.6926930
time: 8.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6927348, upper bound: 154.6926937
time: 9.59 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5453755, upper bound: 154.5453661
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5453755, upper bound: 154.5453661
time: 7.42 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5973225, upper bound: 154.5973142
time: 7.28 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
time: 7.41 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958477, upper bound: 154.5958438
time: 8.65 seconds

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5886085, upper bound: 154.5885953
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5886022, upper bound: 154.5885953
time: 8.13 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993001
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993005
time: 13.91 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
time: 7.43 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6378946, upper bound: 154.6379781
time: 9.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6379008, upper bound: 154.6379758
time: 8.10 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
time: 7.62 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
time: 7.35 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
time: 6.06 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
time: 7.37 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341
time: 6.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 31.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6175353, upper bound: 154.6175276
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6175353, upper bound: 154.6175276
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607411
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6607540, upper bound: 154.6607420
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6927344, upper bound: 154.6926930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6927348, upper bound: 154.6926937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5453755, upper bound: 154.5453661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5453755, upper bound: 154.5453661
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5973225, upper bound: 154.5973142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5958477, upper bound: 154.5958438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5886085, upper bound: 154.5885953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5886022, upper bound: 154.5885953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993001
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6378946, upper bound: 154.6379781
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6379008, upper bound: 154.6379758
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.74
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5871760, upper bound: 154.5871632
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5871760, upper bound: 154.5871632
time: 8.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5453560, upper bound: 154.5453127
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5453560, upper bound: 154.5453127
time: 6.49 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -154.5871760, upper bound: 154.5871632
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -154.5871760, upper bound: 154.5871632
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.20
Output dim: 4, lower bound: -154.5453560, upper bound: 154.5453127
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.20
Output dim: 4, lower bound: -154.5453560, upper bound: 154.5453127
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6607559, upper bound: 154.6607411
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6607540, upper bound: 154.6607420
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6927344, upper bound: 154.6926930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6927348, upper bound: 154.6926937
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5973320, upper bound: 154.5973147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5973225, upper bound: 154.5973142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5901969, upper bound: 154.5901921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5958485, upper bound: 154.5958447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5958477, upper bound: 154.5958438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5886085, upper bound: 154.5885953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5886022, upper bound: 154.5885953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993001
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6992755, upper bound: 154.6993005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6508305, upper bound: 154.6508265
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6378946, upper bound: 154.6379781
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6379008, upper bound: 154.6379758
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6406136, upper bound: 154.6406637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5604562, upper bound: 154.5604875
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.6076265, upper bound: 154.6076495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5938725, upper bound: 154.5938724
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -154.5835176, upper bound: 154.5835341
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6866292, upper bound: 154.6866292
time: 10.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6866292, upper bound: 154.6866292
time: 10.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.90
Output dim: 4, lower bound: -154.6866292, upper bound: 154.6866292
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.90
Output dim: 4, lower bound: -154.6866292, upper bound: 154.6866292

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6833906, upper bound: 154.6833890
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6833890, upper bound: 154.6833906
time: 10.26 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732616
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732616
time: 8.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.84
Output dim: 4, lower bound: -154.6833906, upper bound: 154.6833890
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.84
Output dim: 4, lower bound: -154.6833890, upper bound: 154.6833906
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.84
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732616
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.84
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732616

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6833906, upper bound: 154.6833782
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6833826, upper bound: 154.6833890
time: 8.95 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6693527, upper bound: 154.6693569
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6693527, upper bound: 154.6693569
time: 7.14 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6179759, upper bound: 154.6179790
time: 8.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6179759, upper bound: 154.6179790
time: 16.06 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732608, upper bound: 154.6732616
time: 9.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732608
time: 11.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6833906, upper bound: 154.6833782
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6833826, upper bound: 154.6833890
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6693527, upper bound: 154.6693569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6693527, upper bound: 154.6693569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6179759, upper bound: 154.6179790
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6179759, upper bound: 154.6179790
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6732608, upper bound: 154.6732616
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.42
Output dim: 4, lower bound: -154.6732616, upper bound: 154.6732608

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6832966, upper bound: 154.6832840
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6832958, upper bound: 154.6832845
time: 8.42 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6653081, upper bound: 154.6653280
time: 9.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6653081, upper bound: 154.6653280
time: 10.16 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6667540, upper bound: 154.6667572
time: 8.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6667521, upper bound: 154.6667604
time: 10.55 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6578823, upper bound: 154.6578828
time: 9.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6578823, upper bound: 154.6578828
time: 9.00 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6166532, upper bound: 154.6166660
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6166532, upper bound: 154.6166660
time: 9.44 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6125703, upper bound: 154.6125782
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6125758, upper bound: 154.6125761
time: 10.85 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732608, upper bound: 154.6732593
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732565, upper bound: 154.6732616
time: 7.73 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732147, upper bound: 154.6732135
time: 14.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6732147, upper bound: 154.6732135
time: 9.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6832966, upper bound: 154.6832840
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6832958, upper bound: 154.6832845
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6653081, upper bound: 154.6653280
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6653081, upper bound: 154.6653280
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6667540, upper bound: 154.6667572
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6667521, upper bound: 154.6667604
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6578823, upper bound: 154.6578828
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6578823, upper bound: 154.6578828
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6166532, upper bound: 154.6166660
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6166532, upper bound: 154.6166660
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6125703, upper bound: 154.6125782
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6125758, upper bound: 154.6125761
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6732608, upper bound: 154.6732593
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6732565, upper bound: 154.6732616
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6732147, upper bound: 154.6732135
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.24
Output dim: 4, lower bound: -154.6732147, upper bound: 154.6732135

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6388447, upper bound: 154.6388436
time: 11.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6388447, upper bound: 154.6388436
time: 10.93 seconds

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
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6832958, upper bound: 154.6832845
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6832957, upper bound: 154.6832845
time: 8.85 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6609037, upper bound: 154.6609221
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6609037, upper bound: 154.6609256
time: 8.12 seconds

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6635671, upper bound: 154.6635883
time: 9.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6635711, upper bound: 154.6635796
time: 7.52 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6667540, upper bound: 154.6667509
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6667474, upper bound: 154.6667572
time: 8.12 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6549306, upper bound: 154.6549306
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6549306, upper bound: 154.6549306
time: 9.42 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6555045, upper bound: 154.6555035
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6555046, upper bound: 154.6555031
time: 11.07 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6549306, upper bound: 154.6549306
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6549306, upper bound: 154.6549306
time: 7.67 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958791, upper bound: 154.5958630
time: 9.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5958791, upper bound: 154.5958630
time: 10.13 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6111316, upper bound: 154.6111373
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6111315, upper bound: 154.6111373
time: 7.87 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6098466, upper bound: 154.6098498
time: 10.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6098466, upper bound: 154.6098503
time: 8.61 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6125696, upper bound: 154.6125762
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6125758, upper bound: 154.6125722
time: 6.27 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7149282, upper bound: 154.7149282
time: 11.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7149282, upper bound: 154.7149282
time: 9.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.01
Output dim: 4, lower bound: -154.7149282, upper bound: 154.7149282
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.01
Output dim: 4, lower bound: -154.7149282, upper bound: 154.7149282

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6644593, upper bound: 154.6644593
time: 10.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6644593, upper bound: 154.6644593
time: 11.03 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7148119, upper bound: 154.7148128
time: 11.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7148119, upper bound: 154.7148128
time: 11.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.08
Output dim: 4, lower bound: -154.6644593, upper bound: 154.6644593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.08
Output dim: 4, lower bound: -154.6644593, upper bound: 154.6644593
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.08
Output dim: 4, lower bound: -154.7148119, upper bound: 154.7148128
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.08
Output dim: 4, lower bound: -154.7148119, upper bound: 154.7148128

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564192
time: 14.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564192
time: 10.90 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6408998, upper bound: 154.6408998
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6408998, upper bound: 154.6408998
time: 11.71 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955778
time: 9.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955778
time: 11.06 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023607
time: 11.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023607
time: 11.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564192
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564192
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6408998, upper bound: 154.6408998
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6408998, upper bound: 154.6408998
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955778
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955778
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023607
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023607

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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564180, upper bound: 154.6564192
time: 11.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564180
time: 10.40 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6431842, upper bound: 154.6431777
time: 10.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6431842, upper bound: 154.6431777
time: 9.94 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337808
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337808
time: 11.01 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6342385, upper bound: 154.6342404
time: 11.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6342385, upper bound: 154.6342404
time: 10.74 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955772
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6955770, upper bound: 154.6955778
time: 11.59 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6949869, upper bound: 154.6949896
time: 11.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6949869, upper bound: 154.6949896
time: 11.00 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023600, upper bound: 154.6023605
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023598, upper bound: 154.6023607
time: 5.85 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023604
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6023597, upper bound: 154.6023607
time: 17.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6564180, upper bound: 154.6564192
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6564192, upper bound: 154.6564180
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6431842, upper bound: 154.6431777
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6431842, upper bound: 154.6431777
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337808
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337808
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6342385, upper bound: 154.6342404
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6342385, upper bound: 154.6342404
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6955778, upper bound: 154.6955772
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6955770, upper bound: 154.6955778
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6949869, upper bound: 154.6949896
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6949869, upper bound: 154.6949896
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6023600, upper bound: 154.6023605
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6023598, upper bound: 154.6023607
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023604
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.97
Output dim: 4, lower bound: -154.6023597, upper bound: 154.6023607

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564169, upper bound: 154.6564168
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6564160, upper bound: 154.6564181
time: 11.08 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6285201, upper bound: 154.6285215
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6285201, upper bound: 154.6285215
time: 14.62 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6046964, upper bound: 154.6046964
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6046964, upper bound: 154.6046964
time: 8.34 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6397761, upper bound: 154.6397765
time: 16.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6397768, upper bound: 154.6397751
time: 11.22 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6321647, upper bound: 154.6321652
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6321647, upper bound: 154.6321652
time: 6.75 seconds

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
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337779, upper bound: 154.6337808
time: 10.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337794
time: 9.61 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6336629, upper bound: 154.6336638
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6336622, upper bound: 154.6336649
time: 7.43 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326401, upper bound: 154.6326430
time: 10.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326401, upper bound: 154.6326430
time: 10.76 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6925300, upper bound: 154.6925309
time: 8.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6925300, upper bound: 154.6925309
time: 11.77 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5723384, upper bound: 154.5723352
time: 10.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5723384, upper bound: 154.5723352
time: 12.61 seconds

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6916242, upper bound: 154.6916254
time: 11.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6916242, upper bound: 154.6916258
time: 9.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6564169, upper bound: 154.6564168
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6564160, upper bound: 154.6564181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6285201, upper bound: 154.6285215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6285201, upper bound: 154.6285215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6046964, upper bound: 154.6046964
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6046964, upper bound: 154.6046964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6397761, upper bound: 154.6397765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6397768, upper bound: 154.6397751
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6321647, upper bound: 154.6321652
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6321647, upper bound: 154.6321652
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6337779, upper bound: 154.6337808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6337788, upper bound: 154.6337794
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6336629, upper bound: 154.6336638
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6336622, upper bound: 154.6336649
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6326401, upper bound: 154.6326430
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6326401, upper bound: 154.6326430
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6925300, upper bound: 154.6925309
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6925300, upper bound: 154.6925309
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.5723384, upper bound: 154.5723352
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.5723384, upper bound: 154.5723352
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6916242, upper bound: 154.6916254
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 4, lower bound: -154.6916242, upper bound: 154.6916258
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -154.6949869, upper bound: 154.6949896
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -154.6023600, upper bound: 154.6023605
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -154.6023598, upper bound: 154.6023607
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -154.6023599, upper bound: 154.6023604
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -154.6023597, upper bound: 154.6023607
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1820.21 seconds
