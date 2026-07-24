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
execution time: IAR + LP analysis = 1.08 + 8.88 = 9.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -242.0420621, upper bound: 242.0420621


# Binary Search by BASE starts (time budget: 2690.04 seconds, max iter: 100)

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
Binary search time: 33.22 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2656.82 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
time: 5.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.88 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 10.88
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 10.88
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0419762, upper bound: 242.0419762
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0419762, upper bound: 242.0419762
time: 5.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.66
Output dim: 7, lower bound: -242.0419762, upper bound: 242.0419762
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.66
Output dim: 7, lower bound: -242.0419762, upper bound: 242.0419762

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0413239, upper bound: 242.0412806
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0412806, upper bound: 242.0413239
time: 5.61 seconds

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847
time: 4.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 7, lower bound: -242.0413239, upper bound: 242.0412806
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 7, lower bound: -242.0412806, upper bound: 242.0413239
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847

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
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0275282, upper bound: 242.0274732
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0275282, upper bound: 242.0274732
time: 5.87 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0365736, upper bound: 242.0366086
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0365736, upper bound: 242.0366086
time: 6.81 seconds

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
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9811663, upper bound: 241.9811788
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9811663, upper bound: 241.9811788
time: 4.92 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189738, upper bound: 242.0189845
time: 6.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0275282, upper bound: 242.0274732
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0275282, upper bound: 242.0274732
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0365736, upper bound: 242.0366086
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0365736, upper bound: 242.0366086
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 13.17
Output dim: 7, lower bound: -241.9811663, upper bound: 241.9811788
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 13.17
Output dim: 7, lower bound: -241.9811663, upper bound: 241.9811788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0189748, upper bound: 242.0189847
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 7, lower bound: -242.0189738, upper bound: 242.0189845

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0240266, upper bound: 242.0240067
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0240266, upper bound: 242.0240067
time: 5.42 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9959523, upper bound: 241.9959698
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9959523, upper bound: 241.9959698
time: 6.13 seconds

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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0315681, upper bound: 242.0315723
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0315681, upper bound: 242.0315723
time: 6.14 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318682, upper bound: 242.0319120
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318682, upper bound: 242.0319120
time: 6.47 seconds

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0114577, upper bound: 242.0115014
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0114577, upper bound: 242.0115014
time: 7.18 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0185666, upper bound: 242.0185716
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0185666, upper bound: 242.0185716
time: 5.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0240266, upper bound: 242.0240067
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0240266, upper bound: 242.0240067
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.86
Output dim: 7, lower bound: -241.9959523, upper bound: 241.9959698
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.86
Output dim: 7, lower bound: -241.9959523, upper bound: 241.9959698
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0315681, upper bound: 242.0315723
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0315681, upper bound: 242.0315723
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0318682, upper bound: 242.0319120
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0318682, upper bound: 242.0319120
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0114577, upper bound: 242.0115014
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0114577, upper bound: 242.0115014
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0185666, upper bound: 242.0185716
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.86
Output dim: 7, lower bound: -242.0185666, upper bound: 242.0185716

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
time: 6.92 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233274
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0233447, upper bound: 242.0232634
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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0286152, upper bound: 242.0285500
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0285340, upper bound: 242.0286154
time: 5.26 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0097109, upper bound: 242.0096716
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0097109, upper bound: 242.0096716
time: 5.52 seconds

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
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311570
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311570
time: 7.13 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0184030
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0184030
time: 5.62 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9870813, upper bound: 241.9870959
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9870813, upper bound: 241.9870959
time: 5.31 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0145404, upper bound: 242.0145730
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0145574, upper bound: 242.0145570
time: 5.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0233447, upper bound: 242.0232634
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0286152, upper bound: 242.0285500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0285340, upper bound: 242.0286154
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0097109, upper bound: 242.0096716
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0097109, upper bound: 242.0096716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311570
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0184030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0184030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 11.61
Output dim: 7, lower bound: -241.9870813, upper bound: 241.9870959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 11.61
Output dim: 7, lower bound: -241.9870813, upper bound: 241.9870959
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0145404, upper bound: 242.0145730
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.61
Output dim: 7, lower bound: -242.0145574, upper bound: 242.0145570

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149950
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
time: 6.26 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149547
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0149622, upper bound: 242.0149951
time: 5.72 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232681, upper bound: 242.0233274
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233215
time: 6.10 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232117, upper bound: 242.0231996
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232611, upper bound: 242.0231572
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9823367, upper bound: 241.9823006
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9823367, upper bound: 241.9823006
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144429, upper bound: 242.0146332
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144429, upper bound: 242.0146332
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177539
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177539
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311428
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0311195, upper bound: 242.0311570
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0183136, upper bound: 242.0184030
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0183362
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9815663, upper bound: 241.9815340
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9815663, upper bound: 241.9815340
time: 6.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9953548, upper bound: 241.9953526
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9953548, upper bound: 241.9953526
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9951853, upper bound: 241.9951187
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9951853, upper bound: 241.9951187
time: 5.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149950
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0150050, upper bound: 242.0149547
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0149622, upper bound: 242.0149951
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0232681, upper bound: 242.0233274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0232117, upper bound: 242.0231996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0232611, upper bound: 242.0231572
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9823367, upper bound: 241.9823006
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9823367, upper bound: 241.9823006
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0144429, upper bound: 242.0146332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0144429, upper bound: 242.0146332
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0311195, upper bound: 242.0311570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0183136, upper bound: 242.0184030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.24
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0183362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9815663, upper bound: 241.9815340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9815663, upper bound: 241.9815340
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9953548, upper bound: 241.9953526
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9953548, upper bound: 241.9953526
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9951853, upper bound: 241.9951187
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.24
Output dim: 7, lower bound: -241.9951853, upper bound: 241.9951187

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0122573, upper bound: 242.0122612
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0122573, upper bound: 242.0122612
time: 6.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0117771, upper bound: 242.0117592
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0117771, upper bound: 242.0117592
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0147477, upper bound: 242.0147268
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0147641, upper bound: 242.0147113
time: 6.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0146767, upper bound: 242.0147136
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0146814, upper bound: 242.0146978
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0207159, upper bound: 242.0206339
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0206452, upper bound: 242.0207474
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233209
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232692, upper bound: 242.0233215
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232117, upper bound: 242.0231837
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0231729, upper bound: 242.0231996
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0229468, upper bound: 242.0228590
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0229481, upper bound: 242.0228595
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0141925, upper bound: 242.0143221
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0141907, upper bound: 242.0143227
time: 5.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0038221, upper bound: 242.0039349
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0038221, upper bound: 242.0039349
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177280
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177350, upper bound: 242.0177539
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177280
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177350, upper bound: 242.0177539
time: 6.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 13.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0122573, upper bound: 242.0122612
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0122573, upper bound: 242.0122612
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0117771, upper bound: 242.0117592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0117771, upper bound: 242.0117592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0147477, upper bound: 242.0147268
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0147641, upper bound: 242.0147113
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0146767, upper bound: 242.0147136
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0146814, upper bound: 242.0146978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0207159, upper bound: 242.0206339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0206452, upper bound: 242.0207474
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0232708, upper bound: 242.0233209
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0232692, upper bound: 242.0233215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0232117, upper bound: 242.0231837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0231729, upper bound: 242.0231996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0229468, upper bound: 242.0228590
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0229481, upper bound: 242.0228595
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0141925, upper bound: 242.0143221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0141907, upper bound: 242.0143227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0038221, upper bound: 242.0039349
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0038221, upper bound: 242.0039349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177280
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0177350, upper bound: 242.0177539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0177552, upper bound: 242.0177280
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.62
Output dim: 7, lower bound: -242.0177350, upper bound: 242.0177539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.62
Output dim: 7, lower bound: -242.0311283, upper bound: 242.0311428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.62
Output dim: 7, lower bound: -242.0311195, upper bound: 242.0311570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.62
Output dim: 7, lower bound: -242.0183136, upper bound: 242.0184030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.62
Output dim: 7, lower bound: -242.0183733, upper bound: 242.0183362
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=243.70181274414062
rel_dist={7: [-242.0419761633516, 242.04197616010464]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0419145, upper bound: 242.0418891
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0418891, upper bound: 242.0419145
time: 5.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.42
Output dim: 7, lower bound: -242.0419145, upper bound: 242.0418891
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.42
Output dim: 7, lower bound: -242.0418891, upper bound: 242.0419145

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0417082, upper bound: 242.0417038
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0417082, upper bound: 242.0417036
time: 7.08 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384845
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384845
time: 5.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 11.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.83
Output dim: 7, lower bound: -242.0417082, upper bound: 242.0417038
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.83
Output dim: 7, lower bound: -242.0417082, upper bound: 242.0417036
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.83
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384845
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.83
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384845

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0249682, upper bound: 242.0249862
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0249682, upper bound: 242.0249862
time: 5.02 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393750
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0393787, upper bound: 242.0393558
time: 5.36 seconds

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
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384476
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384183, upper bound: 242.0384845
time: 6.42 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0372741, upper bound: 242.0373413
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0373072, upper bound: 242.0373210
time: 6.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0249682, upper bound: 242.0249862
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0249682, upper bound: 242.0249862
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393750
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0393787, upper bound: 242.0393558
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0384454, upper bound: 242.0384476
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0384183, upper bound: 242.0384845
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0372741, upper bound: 242.0373413
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.57
Output dim: 7, lower bound: -242.0373072, upper bound: 242.0373210

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9895699, upper bound: 241.9896443
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9895699, upper bound: 241.9896443
time: 4.43 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0068726, upper bound: 242.0069085
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0068726, upper bound: 242.0069085
time: 5.42 seconds

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393749
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393750
time: 5.85 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350407, upper bound: 242.0349277
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350407, upper bound: 242.0349277
time: 5.87 seconds

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
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368303
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368303
time: 6.39 seconds

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
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384177, upper bound: 242.0384839
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0384183, upper bound: 242.0384845
time: 6.93 seconds

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
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0357848, upper bound: 242.0358243
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0357848, upper bound: 242.0358243
time: 6.55 seconds

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
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0339244, upper bound: 242.0338732
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0339247, upper bound: 242.0338695
time: 5.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.80
Output dim: 7, lower bound: -241.9895699, upper bound: 241.9896443
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.80
Output dim: 7, lower bound: -241.9895699, upper bound: 241.9896443
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0068726, upper bound: 242.0069085
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0068726, upper bound: 242.0069085
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393749
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0393581, upper bound: 242.0393750
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0350407, upper bound: 242.0349277
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0350407, upper bound: 242.0349277
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368303
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368303
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0384177, upper bound: 242.0384839
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0384183, upper bound: 242.0384845
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0357848, upper bound: 242.0358243
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0357848, upper bound: 242.0358243
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0339244, upper bound: 242.0338732
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.80
Output dim: 7, lower bound: -242.0339247, upper bound: 242.0338695

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318485
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318485
time: 6.04 seconds

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0225569, upper bound: 242.0225882
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0225569, upper bound: 242.0225882
time: 6.07 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9869817, upper bound: 241.9869678
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9869817, upper bound: 241.9869678
time: 5.53 seconds

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0347919, upper bound: 242.0347765
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0348373, upper bound: 242.0347458
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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0227879, upper bound: 242.0228191
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0227879, upper bound: 242.0228191
time: 7.42 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368468, upper bound: 242.0368303
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368299
time: 6.51 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0379641, upper bound: 242.0380233
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0379694, upper bound: 242.0380212
time: 5.91 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0377639, upper bound: 242.0378229
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0377521, upper bound: 242.0378291
time: 6.03 seconds

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
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351270
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351267
time: 6.11 seconds

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
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351270
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351267
time: 6.09 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9764630, upper bound: 241.9764694
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9764630, upper bound: 241.9764694
time: 4.19 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477
time: 5.24 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.66 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318485
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318485
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0225569, upper bound: 242.0225882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0225569, upper bound: 242.0225882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.66
Output dim: 7, lower bound: -241.9869817, upper bound: 241.9869678
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.66
Output dim: 7, lower bound: -241.9869817, upper bound: 241.9869678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0347919, upper bound: 242.0347765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0348373, upper bound: 242.0347458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0227879, upper bound: 242.0228191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0227879, upper bound: 242.0228191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0368468, upper bound: 242.0368303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0368474, upper bound: 242.0368299
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0379641, upper bound: 242.0380233
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0379694, upper bound: 242.0380212
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0377639, upper bound: 242.0378229
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0377521, upper bound: 242.0378291
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351270
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351267
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.66
Output dim: 7, lower bound: -241.9764630, upper bound: 241.9764694
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.66
Output dim: 7, lower bound: -241.9764630, upper bound: 241.9764694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.66
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0317093, upper bound: 242.0316876
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0317058, upper bound: 242.0316902
time: 6.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318686, upper bound: 242.0318485
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318483
time: 7.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9363014, upper bound: 241.9362849
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9363014, upper bound: 241.9362849
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9830797, upper bound: 241.9830756
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9830797, upper bound: 241.9830756
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0347883, upper bound: 242.0347765
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0347919, upper bound: 242.0347668
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9982478, upper bound: 241.9981253
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9982478, upper bound: 241.9981253
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0042875, upper bound: 242.0042931
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0042875, upper bound: 242.0042931
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0140846, upper bound: 242.0140833
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0140846, upper bound: 242.0140833
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368023, upper bound: 242.0368303
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0368468, upper bound: 242.0367906
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0338147, upper bound: 242.0338111
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0338132, upper bound: 242.0338177
time: 6.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0337007, upper bound: 242.0337546
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0337007, upper bound: 242.0337546
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0364592, upper bound: 242.0364681
time: 8.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0364592, upper bound: 242.0364681
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0326635, upper bound: 242.0326927
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0326635, upper bound: 242.0326927
time: 7.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0374981, upper bound: 242.0375738
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0374981, upper bound: 242.0375739
time: 5.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0111074, upper bound: 242.0111625
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0111074, upper bound: 242.0111625
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350709, upper bound: 242.0351267
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351217
time: 6.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351113
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0350764, upper bound: 242.0351270
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0227824, upper bound: 242.0228319
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0227824, upper bound: 242.0228319
time: 6.45 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0317093, upper bound: 242.0316876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0317058, upper bound: 242.0316902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0318686, upper bound: 242.0318485
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0318711, upper bound: 242.0318483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9363014, upper bound: 241.9362849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9363014, upper bound: 241.9362849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9830797, upper bound: 241.9830756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9830797, upper bound: 241.9830756
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0347883, upper bound: 242.0347765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0347919, upper bound: 242.0347668
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9982478, upper bound: 241.9981253
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -241.9982478, upper bound: 241.9981253
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0042875, upper bound: 242.0042931
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0042875, upper bound: 242.0042931
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0140846, upper bound: 242.0140833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0140846, upper bound: 242.0140833
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0368023, upper bound: 242.0368303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0368468, upper bound: 242.0367906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0338147, upper bound: 242.0338111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0338132, upper bound: 242.0338177
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0337007, upper bound: 242.0337546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0337007, upper bound: 242.0337546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0364592, upper bound: 242.0364681
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0364592, upper bound: 242.0364681
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0326635, upper bound: 242.0326927
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0326635, upper bound: 242.0326927
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0374981, upper bound: 242.0375738
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0374981, upper bound: 242.0375739
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0111074, upper bound: 242.0111625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0111074, upper bound: 242.0111625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0350709, upper bound: 242.0351267
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0350933, upper bound: 242.0351217
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0350938, upper bound: 242.0351113
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0350764, upper bound: 242.0351270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0227824, upper bound: 242.0228319
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.77
Output dim: 7, lower bound: -242.0227824, upper bound: 242.0228319
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.77
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.77
Output dim: 7, lower bound: -242.0325805, upper bound: 242.0325477
Binary search (step 2): status=Status.UNKNOWN, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=243.70181274414062
rel_dist={7: [-242.0419144615069, 242.04191446102539]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1232.65 seconds
