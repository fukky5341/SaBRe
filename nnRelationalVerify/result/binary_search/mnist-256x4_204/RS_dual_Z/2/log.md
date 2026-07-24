## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 242.013661301
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635)
1: (-109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755)
2: (-144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974)
3: (-153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142)
4: (-141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818)
5: (-126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261)
6: (-121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238)
7: (-132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127)
8: (-158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799)
9: (-120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 8.74 = 9.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -242.0420621, upper bound: 242.0420621


# Binary Search by BASE starts (time budget: 2690.17 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=243.70181274414062
rel_dist={7: [-242.0414140657913, 242.04141406382126]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=243.70181274414062
rel_dist={7: [-242.04091628504267, 242.04091628504267]}

## Binary Search Result
Binary search time: 32.75 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2657.42 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0407327, upper bound: 242.0407536
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0407536, upper bound: 242.0407327
time: 6.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.94
Output dim: 7, lower bound: -242.0407327, upper bound: 242.0407536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.94
Output dim: 7, lower bound: -242.0407536, upper bound: 242.0407327

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0223517, upper bound: 242.0223779
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0223517, upper bound: 242.0223779
time: 6.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0223779, upper bound: 242.0223517
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0223779, upper bound: 242.0223517
time: 6.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.22
Output dim: 7, lower bound: -242.0223517, upper bound: 242.0223779
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.22
Output dim: 7, lower bound: -242.0223517, upper bound: 242.0223779
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.22
Output dim: 7, lower bound: -242.0223779, upper bound: 242.0223517
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.22
Output dim: 7, lower bound: -242.0223779, upper bound: 242.0223517

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177409, upper bound: 242.0177711
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177449, upper bound: 242.0177666
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177409, upper bound: 242.0177711
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177449, upper bound: 242.0177666
time: 8.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177666, upper bound: 242.0177449
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177711, upper bound: 242.0177409
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177666, upper bound: 242.0177449
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177711, upper bound: 242.0177409
time: 6.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177409, upper bound: 242.0177711
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177449, upper bound: 242.0177666
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177409, upper bound: 242.0177711
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177449, upper bound: 242.0177666
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177666, upper bound: 242.0177449
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177711, upper bound: 242.0177409
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177666, upper bound: 242.0177449
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -242.0177711, upper bound: 242.0177409

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 5.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.92
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 5.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170329
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 7.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169859, upper bound: 242.0170322
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
time: 6.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170329
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169859, upper bound: 242.0170322
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
time: 6.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 6.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
time: 6.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
time: 7.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
time: 7.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170329
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169859, upper bound: 242.0170322
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169859, upper bound: 242.0170322
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.15
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139847, upper bound: 242.0139650
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140041
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139847, upper bound: 242.0139650
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140041
time: 6.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139809, upper bound: 242.0139799
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139614, upper bound: 242.0140105
time: 6.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139809, upper bound: 242.0139799
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139614, upper bound: 242.0140105
time: 6.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139864, upper bound: 242.0139688
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140042
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139864, upper bound: 242.0139688
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140042
time: 6.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139815, upper bound: 242.0139796
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139604, upper bound: 242.0140107
time: 6.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139847, upper bound: 242.0139650
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140041
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139847, upper bound: 242.0139650
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139809, upper bound: 242.0139799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139614, upper bound: 242.0140105
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139809, upper bound: 242.0139799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139614, upper bound: 242.0140105
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139864, upper bound: 242.0139688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140042
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139864, upper bound: 242.0139688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139698, upper bound: 242.0140042
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139815, upper bound: 242.0139796
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.90
Output dim: 7, lower bound: -242.0139604, upper bound: 242.0140107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170035, upper bound: 242.0170126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0169862, upper bound: 242.0170329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170063, upper bound: 242.0170136
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0169859, upper bound: 242.0170322
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0169861, upper bound: 242.0170322
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170322, upper bound: 242.0169859
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170136, upper bound: 242.0170063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170329, upper bound: 242.0169862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170331, upper bound: 242.0169862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 7, lower bound: -242.0170126, upper bound: 242.0170035
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0401617, upper bound: 242.0401861
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0401861, upper bound: 242.0401617
time: 8.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.49
Output dim: 7, lower bound: -242.0401617, upper bound: 242.0401861
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.49
Output dim: 7, lower bound: -242.0401861, upper bound: 242.0401617

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0215966, upper bound: 242.0216126
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0215966, upper bound: 242.0216126
time: 8.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0216126, upper bound: 242.0215966
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0216126, upper bound: 242.0215966
time: 6.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.07
Output dim: 7, lower bound: -242.0215966, upper bound: 242.0216126
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.07
Output dim: 7, lower bound: -242.0215966, upper bound: 242.0216126
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.07
Output dim: 7, lower bound: -242.0216126, upper bound: 242.0215966
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.07
Output dim: 7, lower bound: -242.0216126, upper bound: 242.0215966

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170696, upper bound: 242.0171078
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170795, upper bound: 242.0170980
time: 8.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170696, upper bound: 242.0171078
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170795, upper bound: 242.0170980
time: 7.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170980, upper bound: 242.0170795
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0171078, upper bound: 242.0170696
time: 7.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170980, upper bound: 242.0170795
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0171078, upper bound: 242.0170696
time: 8.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170696, upper bound: 242.0171078
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170795, upper bound: 242.0170980
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170696, upper bound: 242.0171078
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170795, upper bound: 242.0170980
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170980, upper bound: 242.0170795
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0171078, upper bound: 242.0170696
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0170980, upper bound: 242.0170795
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -242.0171078, upper bound: 242.0170696

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164162
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164238
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164101
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164140
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164162
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164238
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164101
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164140
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163880
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163943
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163829
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163844
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163880
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163943
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163829
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163844
time: 6.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164162
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164238
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164101
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164140
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164162
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164238
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164101
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164140
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163880
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163943
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163829
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163844
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163880
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163943
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163829
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163844

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163833, upper bound: 242.0164162
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164105
time: 6.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163819, upper bound: 242.0164238
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164158
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
time: 5.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
time: 7.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163833, upper bound: 242.0164162
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164105
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163819, upper bound: 242.0164238
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164158
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
time: 7.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
time: 6.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833
time: 7.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
time: 6.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833
time: 8.24 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163833, upper bound: 242.0164162
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164105
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163819, upper bound: 242.0164238
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163833, upper bound: 242.0164162
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164105
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163819, upper bound: 242.0164238
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.97
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134615, upper bound: 242.0134509
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134566, upper bound: 242.0134661
time: 8.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134615, upper bound: 242.0134507
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134569, upper bound: 242.0134650
time: 7.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134592, upper bound: 242.0134579
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134514, upper bound: 242.0134706
time: 7.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134592, upper bound: 242.0134574
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0134515, upper bound: 242.0134700
time: 6.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134615, upper bound: 242.0134509
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134566, upper bound: 242.0134661
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134615, upper bound: 242.0134507
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134569, upper bound: 242.0134650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134592, upper bound: 242.0134579
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134514, upper bound: 242.0134706
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134592, upper bound: 242.0134574
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.47
Output dim: 7, lower bound: -242.0134515, upper bound: 242.0134700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163833, upper bound: 242.0164162
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163844, upper bound: 242.0164105
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163819, upper bound: 242.0164238
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163829, upper bound: 242.0164158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163908, upper bound: 242.0164101
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163943, upper bound: 242.0164066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163871, upper bound: 242.0164140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0163880, upper bound: 242.0164096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164096, upper bound: 242.0163880
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164140, upper bound: 242.0163871
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164066, upper bound: 242.0163943
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164101, upper bound: 242.0163908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164158, upper bound: 242.0163829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164238, upper bound: 242.0163819
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164105, upper bound: 242.0163844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.47
Output dim: 7, lower bound: -242.0164162, upper bound: 242.0163833
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=243.70181274414062
rel_dist={7: [-242.0414140657913, 242.04141406382126]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0395898, upper bound: 242.0395983
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0395983, upper bound: 242.0395898
time: 10.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 7, lower bound: -242.0395898, upper bound: 242.0395983
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 7, lower bound: -242.0395983, upper bound: 242.0395898

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0209830, upper bound: 242.0209891
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0209830, upper bound: 242.0209891
time: 6.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0209891, upper bound: 242.0209830
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0209891, upper bound: 242.0209830
time: 7.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.82
Output dim: 7, lower bound: -242.0209830, upper bound: 242.0209891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.82
Output dim: 7, lower bound: -242.0209830, upper bound: 242.0209891
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.82
Output dim: 7, lower bound: -242.0209891, upper bound: 242.0209830
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.82
Output dim: 7, lower bound: -242.0209891, upper bound: 242.0209830

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163926
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163894
time: 9.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163926
time: 8.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163894
time: 9.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163894, upper bound: 242.0163797
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163926, upper bound: 242.0163773
time: 7.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163894, upper bound: 242.0163797
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0163926, upper bound: 242.0163773
time: 7.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163926
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163894
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163926
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163773, upper bound: 242.0163894
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163894, upper bound: 242.0163797
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163926, upper bound: 242.0163773
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163894, upper bound: 242.0163797
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.10
Output dim: 7, lower bound: -242.0163926, upper bound: 242.0163773

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156971
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156997
time: 6.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156946
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156955
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156971
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156997
time: 6.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156946
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156955
time: 9.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156862
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156875
time: 6.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156838
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156846
time: 8.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156862
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156875
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156838
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156846
time: 8.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156971
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156997
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156946
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156955
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156971
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156997
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156946
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156955
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156862
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156875
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156838
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156846
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156862
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156875
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156838
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.59
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156846

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156971
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156959
time: 8.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156971
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156959
time: 8.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
time: 5.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156861
time: 7.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
time: 7.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156861
time: 9.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
time: 7.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
time: 7.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843
time: 7.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156971
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156861
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0127826, upper bound: 242.0127804
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0127801, upper bound: 242.0127870
time: 8.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635
1: -109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755
2: -144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974
3: -153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142
4: -141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818
5: -126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261
6: -121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238
7: -132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127
8: -158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799
9: -120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0127826, upper bound: 242.0127804
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0127801, upper bound: 242.0127870
time: 9.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 7, lower bound: -242.0127826, upper bound: 242.0127804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 7, lower bound: -242.0127801, upper bound: 242.0127870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 7, lower bound: -242.0127826, upper bound: 242.0127804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 7, lower bound: -242.0127801, upper bound: 242.0127870
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156846, upper bound: 242.0156959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156836, upper bound: 242.0156997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156838, upper bound: 242.0156989
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156869, upper bound: 242.0156946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156875, upper bound: 242.0156942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156861, upper bound: 242.0156955
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156862, upper bound: 242.0156951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156955, upper bound: 242.0156861
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156951, upper bound: 242.0156862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156843, upper bound: 242.0156861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156942, upper bound: 242.0156875
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156946, upper bound: 242.0156869
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156989, upper bound: 242.0156838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156997, upper bound: 242.0156836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156959, upper bound: 242.0156846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 7, lower bound: -242.0156971, upper bound: 242.0156843
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=243.70181274414062
rel_dist={7: [-242.04091628504267, 242.04091628504267]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1815.65 seconds
