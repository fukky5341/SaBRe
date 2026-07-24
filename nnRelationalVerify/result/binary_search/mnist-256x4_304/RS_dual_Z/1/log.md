## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 173.89956106530002
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 9.62 = 10.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0741399, upper bound: 174.0741399


# Binary Search by BASE starts (time budget: 1989.14 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search Result
Binary search time: 39.00 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1950.14 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072
time: 6.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.14
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.14
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0488921, upper bound: 174.0489072
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0488921
time: 7.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0488921, upper bound: 174.0489072
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0488920
time: 8.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.02
Output dim: 7, lower bound: -174.0488921, upper bound: 174.0489072
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.02
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0488921
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.02
Output dim: 7, lower bound: -174.0488921, upper bound: 174.0489072
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.02
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0488920

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
time: 9.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483590
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
time: 8.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483363
time: 8.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483590
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483363, upper bound: 174.0483591
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483361
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.06
Output dim: 7, lower bound: -174.0483591, upper bound: 174.0483363

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 6.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 8.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
time: 6.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050929, upper bound: 174.0050974
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -174.0050974, upper bound: 174.0050929

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 7.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 7.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 6.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 6.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 6.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
time: 6.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025101, upper bound: 174.0025115
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025136, upper bound: 174.0025080
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025059, upper bound: 174.0025198
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025079, upper bound: 174.0025157
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025101, upper bound: 174.0025115
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0025136, upper bound: 174.0025080
time: 6.19 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025101, upper bound: 174.0025115
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025136, upper bound: 174.0025080
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025059, upper bound: 174.0025198
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025079, upper bound: 174.0025157
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025101, upper bound: 174.0025115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.18
Output dim: 7, lower bound: -174.0025136, upper bound: 174.0025080
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050861, upper bound: 174.0050850
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050805, upper bound: 174.0050906
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050906, upper bound: 174.0050805
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 7, lower bound: -174.0050850, upper bound: 174.0050861
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 7.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.13
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.13
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484529
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
time: 7.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484527
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
time: 7.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484529
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484527
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
time: 8.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
time: 8.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.53
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 7.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 7.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 8.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 8.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.99
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.24 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480803, upper bound: 174.0480803
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480803, upper bound: 174.0480803
time: 8.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.44
Output dim: 7, lower bound: -174.0480803, upper bound: 174.0480803
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.44
Output dim: 7, lower bound: -174.0480803, upper bound: 174.0480803

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480779, upper bound: 174.0480803
time: 9.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480778
time: 8.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480779, upper bound: 174.0480801
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480780
time: 9.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.97 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -174.0480779, upper bound: 174.0480803
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480778
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -174.0480779, upper bound: 174.0480801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480780

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
time: 10.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
time: 9.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
time: 10.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
time: 9.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474896, upper bound: 174.0474938
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.26
Output dim: 7, lower bound: -174.0474938, upper bound: 174.0474896

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 8.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
time: 8.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
time: 8.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044147, upper bound: 174.0044165
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
time: 9.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
time: 9.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
time: 9.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
time: 10.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044087
time: 10.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044077
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044087
time: 9.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044077
time: 9.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
time: 9.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 10.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 10.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044087
time: 10.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
time: 9.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
time: 9.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
time: 9.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044087
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044087
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044077
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044087
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044087
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044087, upper bound: 174.0044094
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044084, upper bound: 174.0044091
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044097, upper bound: 174.0044077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.13
Output dim: 7, lower bound: -174.0044094, upper bound: 174.0044080
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -174.0044158, upper bound: 174.0044154
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1828.51 seconds
