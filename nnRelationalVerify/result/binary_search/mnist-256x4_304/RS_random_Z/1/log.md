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
execution time: IAR + LP analysis = 1.38 + 9.74 = 11.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0741399, upper bound: 174.0741399


# Binary Search by BASE starts (time budget: 1988.88 seconds, max iter: 100)

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
Binary search time: 38.90 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1949.97 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0333466, upper bound: 174.0333466
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0333466, upper bound: 174.0333466
time: 7.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.91
Output dim: 7, lower bound: -174.0333466, upper bound: 174.0333466
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.91
Output dim: 7, lower bound: -174.0333466, upper bound: 174.0333466

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923867
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923867
time: 6.67 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321532
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321532
time: 7.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923867
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923867
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321532
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321532

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9892077, upper bound: 173.9892044
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9892044, upper bound: 173.9892077
time: 5.28 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923745
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923745, upper bound: 173.9923867
time: 6.75 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321372
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321372, upper bound: 174.0321532
time: 8.54 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321530
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321532
time: 6.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -173.9892077, upper bound: 173.9892044
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -173.9892044, upper bound: 173.9892077
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -173.9923867, upper bound: 173.9923745
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -173.9923745, upper bound: 173.9923867
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321372
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -174.0321372, upper bound: 174.0321532
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321530
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321532

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820102
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820102
time: 5.60 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861774, upper bound: 173.9862010
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861966, upper bound: 173.9861867
time: 7.06 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861671, upper bound: 173.9861668
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861669, upper bound: 173.9861667
time: 6.42 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872240
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872240
time: 6.93 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321527, upper bound: 174.0321372
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321370
time: 7.54 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914170
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914170
time: 6.25 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321526
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321530
time: 5.88 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0136361, upper bound: 174.0136360
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0136361, upper bound: 174.0136360
time: 6.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820102
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820102
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9861774, upper bound: 173.9862010
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9861966, upper bound: 173.9861867
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9861671, upper bound: 173.9861668
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9861669, upper bound: 173.9861667
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872240
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872240
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0321527, upper bound: 174.0321372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321370
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914170
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914170
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0321532, upper bound: 174.0321526
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321530
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0136361, upper bound: 174.0136360
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.52
Output dim: 7, lower bound: -174.0136361, upper bound: 174.0136360

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820069
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820068, upper bound: 173.9820102
time: 7.01 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820064
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820060, upper bound: 173.9820102
time: 7.49 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795856, upper bound: 173.9795971
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795869, upper bound: 173.9795939
time: 6.17 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9810755, upper bound: 173.9810721
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9810755, upper bound: 173.9810721
time: 6.60 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861671, upper bound: 173.9861591
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9861579, upper bound: 173.9861667
time: 6.81 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9238580, upper bound: 173.9238610
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9238580, upper bound: 173.9238610
time: 7.93 seconds

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9042026, upper bound: 173.9042034
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9042026, upper bound: 173.9042034
time: 6.71 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9872059, upper bound: 173.9872240
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872143
time: 6.84 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034173, upper bound: 174.0034131
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034173, upper bound: 174.0034131
time: 6.39 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321529, upper bound: 174.0321370
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321529, upper bound: 174.0321370
time: 7.19 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914091
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914031, upper bound: 173.9914170
time: 7.30 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914169
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9914056, upper bound: 173.9914170
time: 6.61 seconds

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
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0308409, upper bound: 174.0308491
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0308409, upper bound: 174.0308491
time: 6.59 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321466
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0321464, upper bound: 174.0321530
time: 7.03 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9229061, upper bound: 173.9229105
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9229061, upper bound: 173.9229105
time: 6.28 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9592823, upper bound: 173.9592669
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9592823, upper bound: 173.9592669
time: 7.45 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9820068, upper bound: 173.9820102
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9820064
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9820060, upper bound: 173.9820102
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9795856, upper bound: 173.9795971
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9795869, upper bound: 173.9795939
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9810755, upper bound: 173.9810721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9810755, upper bound: 173.9810721
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9861671, upper bound: 173.9861591
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9861579, upper bound: 173.9861667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9238580, upper bound: 173.9238610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9238580, upper bound: 173.9238610
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9042026, upper bound: 173.9042034
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9042026, upper bound: 173.9042034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9872059, upper bound: 173.9872240
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9872121, upper bound: 173.9872143
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0034173, upper bound: 174.0034131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0034173, upper bound: 174.0034131
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0321529, upper bound: 174.0321370
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0321529, upper bound: 174.0321370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914091
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9914031, upper bound: 173.9914170
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9914074, upper bound: 173.9914169
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9914056, upper bound: 173.9914170
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0308409, upper bound: 174.0308491
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0308409, upper bound: 174.0308491
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0321530, upper bound: 174.0321466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -174.0321464, upper bound: 174.0321530
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9229061, upper bound: 173.9229105
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9229061, upper bound: 173.9229105
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9592823, upper bound: 173.9592669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 7, lower bound: -173.9592823, upper bound: 173.9592669

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9819834, upper bound: 173.9820069
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820100, upper bound: 173.9819838
time: 5.80 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794519, upper bound: 173.9794525
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794506, upper bound: 173.9794552
time: 6.54 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9809142, upper bound: 173.9809484
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9809514, upper bound: 173.9809119
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820017, upper bound: 173.9820102
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9820060, upper bound: 173.9820085
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795192, upper bound: 173.9795292
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795193, upper bound: 173.9795221
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795857, upper bound: 173.9795925
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9795869, upper bound: 173.9795939
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9810754, upper bound: 173.9810486
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9810584, upper bound: 173.9810721
time: 7.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0735119, upper bound: 174.0735160
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0735160, upper bound: 174.0735119
time: 7.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.58
Output dim: 7, lower bound: -174.0735119, upper bound: 174.0735160
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.58
Output dim: 7, lower bound: -174.0735160, upper bound: 174.0735119

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164056
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164056
time: 7.34 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0593001, upper bound: 174.0593004
time: 9.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0593001, upper bound: 174.0593004
time: 8.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.99
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164056
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.99
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164056
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.99
Output dim: 7, lower bound: -174.0593001, upper bound: 174.0593004
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.99
Output dim: 7, lower bound: -174.0593001, upper bound: 174.0593004

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164042
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164027, upper bound: 174.0164056
time: 7.85 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164042
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0164027, upper bound: 174.0164056
time: 7.81 seconds

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0570286, upper bound: 174.0570285
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0570286, upper bound: 174.0570285
time: 7.51 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532354, upper bound: 174.0532358
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532354, upper bound: 174.0532358
time: 7.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164042
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0164027, upper bound: 174.0164056
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0164037, upper bound: 174.0164042
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0164027, upper bound: 174.0164056
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0570286, upper bound: 174.0570285
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0570286, upper bound: 174.0570285
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0532354, upper bound: 174.0532358
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 7, lower bound: -174.0532354, upper bound: 174.0532358

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987421
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987421
time: 6.85 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150736
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150736
time: 7.64 seconds

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0040461, upper bound: 174.0040457
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0040461, upper bound: 174.0040457
time: 7.33 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0163989, upper bound: 174.0164021
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0163989, upper bound: 174.0164021
time: 7.27 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0352572, upper bound: 174.0352530
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0352572, upper bound: 174.0352530
time: 7.04 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501571
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501571
time: 6.42 seconds

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532203
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532203
time: 7.41 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0312885, upper bound: 174.0312819
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0312885, upper bound: 174.0312819
time: 6.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987421
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987421
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150736
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150736
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0040461, upper bound: 174.0040457
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0040461, upper bound: 174.0040457
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0163989, upper bound: 174.0164021
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0163989, upper bound: 174.0164021
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0352572, upper bound: 174.0352530
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0352572, upper bound: 174.0352530
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501571
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501571
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532203
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532203
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0312885, upper bound: 174.0312819
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.39
Output dim: 7, lower bound: -174.0312885, upper bound: 174.0312819

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
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987373
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987357, upper bound: 173.9987421
time: 6.62 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9972307, upper bound: 173.9972241
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9972242, upper bound: 173.9972301
time: 6.88 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150716
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150679, upper bound: 174.0150736
time: 7.91 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0115100, upper bound: 174.0115107
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0115069, upper bound: 174.0115135
time: 7.01 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9976811, upper bound: 173.9976759
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9976744, upper bound: 173.9976817
time: 8.63 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
time: 7.34 seconds

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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
time: 7.42 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
time: 6.60 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
time: 7.98 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
time: 7.98 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501528
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9501531, upper bound: 173.9501571
time: 6.73 seconds

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
time: 6.88 seconds

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
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
time: 6.78 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532184
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0532182, upper bound: 174.0532203
time: 7.56 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
time: 6.69 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306
time: 8.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9987357, upper bound: 173.9987421
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9972307, upper bound: 173.9972241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9972242, upper bound: 173.9972301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0150680, upper bound: 174.0150716
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0150679, upper bound: 174.0150736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0115100, upper bound: 174.0115107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0115069, upper bound: 174.0115135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9976811, upper bound: 173.9976759
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9976744, upper bound: 173.9976817
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501528
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9501531, upper bound: 173.9501571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532184
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0532182, upper bound: 174.0532203
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.26
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987331
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987301, upper bound: 173.9987373
time: 6.85 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987347, upper bound: 173.9987421
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9987357, upper bound: 173.9987420
time: 7.56 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9968139, upper bound: 173.9968132
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9968139, upper bound: 173.9968132
time: 6.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9956414, upper bound: 173.9956445
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9956414, upper bound: 173.9956446
time: 7.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150602, upper bound: 174.0150566
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0150551, upper bound: 174.0150638
time: 8.48 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 18.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9987415, upper bound: 173.9987331
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9987301, upper bound: 173.9987373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9987347, upper bound: 173.9987421
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9987357, upper bound: 173.9987420
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9968139, upper bound: 173.9968132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9968139, upper bound: 173.9968132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9956414, upper bound: 173.9956445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -173.9956414, upper bound: 173.9956446
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -174.0150602, upper bound: 174.0150566
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.25
Output dim: 7, lower bound: -174.0150551, upper bound: 174.0150638
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0150679, upper bound: 174.0150736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0115100, upper bound: 174.0115107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0115069, upper bound: 174.0115135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9976811, upper bound: 173.9976759
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9976744, upper bound: 173.9976817
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0037739, upper bound: 174.0037742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0024651, upper bound: 174.0024568
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9428712, upper bound: 173.9428731
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0347515, upper bound: 174.0347456
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9501549, upper bound: 173.9501528
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9501531, upper bound: 173.9501571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9323344, upper bound: 173.9323336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9923361, upper bound: 173.9923340
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0532201, upper bound: 174.0532184
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0532182, upper bound: 174.0532203
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -173.9288096, upper bound: 173.9288078
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 7, lower bound: -174.0261313, upper bound: 174.0261306
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671405, upper bound: 174.0671405
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671405, upper bound: 174.0671411
time: 7.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.41
Output dim: 7, lower bound: -174.0671405, upper bound: 174.0671405
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.41
Output dim: 7, lower bound: -174.0671405, upper bound: 174.0671411

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671411, upper bound: 174.0671393
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671401, upper bound: 174.0671405
time: 8.35 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701
time: 9.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.15
Output dim: 7, lower bound: -174.0671411, upper bound: 174.0671393
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.15
Output dim: 7, lower bound: -174.0671401, upper bound: 174.0671405
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.15
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.15
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671411, upper bound: 174.0671386
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671404, upper bound: 174.0671393
time: 8.07 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9825090, upper bound: 173.9825089
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9825090, upper bound: 173.9825089
time: 8.08 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628699, upper bound: 174.0628701
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701
time: 8.43 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628698, upper bound: 174.0628701
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628698, upper bound: 174.0628701
time: 8.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0671411, upper bound: 174.0671386
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0671404, upper bound: 174.0671393
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -173.9825090, upper bound: 173.9825089
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -173.9825090, upper bound: 173.9825089
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0628699, upper bound: 174.0628701
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628701
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0628698, upper bound: 174.0628701
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.66
Output dim: 7, lower bound: -174.0628698, upper bound: 174.0628701

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0088394, upper bound: 174.0088349
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0088394, upper bound: 174.0088349
time: 8.12 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671270, upper bound: 174.0671269
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0671263, upper bound: 174.0671257
time: 8.88 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9616534, upper bound: 173.9616533
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9616534, upper bound: 173.9616533
time: 7.86 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9398372, upper bound: 173.9398369
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9398372, upper bound: 173.9398369
time: 7.29 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262106
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262106
time: 7.34 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628689
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628701
time: 8.60 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0330072, upper bound: 174.0330063
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0330072, upper bound: 174.0330063
time: 8.67 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628701
time: 9.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628690
time: 8.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0088394, upper bound: 174.0088349
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0088394, upper bound: 174.0088349
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0671270, upper bound: 174.0671269
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0671263, upper bound: 174.0671257
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -173.9616534, upper bound: 173.9616533
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -173.9616534, upper bound: 173.9616533
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -173.9398372, upper bound: 173.9398369
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -173.9398372, upper bound: 173.9398369
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262106
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262106
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628689
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628701
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0330072, upper bound: 174.0330063
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0330072, upper bound: 174.0330063
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0628688, upper bound: 174.0628701
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.74
Output dim: 7, lower bound: -174.0628700, upper bound: 174.0628690

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9743886, upper bound: 173.9743842
time: 53.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9743886, upper bound: 173.9743842
time: 46.07 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0088394, upper bound: 174.0088347
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0088392, upper bound: 174.0088349
time: 7.18 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0486428, upper bound: 174.0486437
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0486428, upper bound: 174.0486437
time: 9.25 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011925, upper bound: 174.0011961
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011925, upper bound: 174.0011961
time: 7.35 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9616524, upper bound: 173.9616533
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9616534, upper bound: 173.9616507
time: 7.15 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9260627, upper bound: 173.9260636
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9260627, upper bound: 173.9260636
time: 8.62 seconds

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
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9398343, upper bound: 173.9398369
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9398372, upper bound: 173.9398332
time: 7.88 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9244798, upper bound: 173.9244789
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9244798, upper bound: 173.9244789
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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262106
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262087, upper bound: 174.0262106
time: 8.30 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262089, upper bound: 174.0262085
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0262072, upper bound: 174.0262106
time: 9.33 seconds

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
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0622612, upper bound: 174.0622588
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0622612, upper bound: 174.0622588
time: 10.37 seconds

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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0568313, upper bound: 174.0568362
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0568313, upper bound: 174.0568362
time: 9.96 seconds

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
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1804.05 seconds
