## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 195.3388952653
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864)
1: (-94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567)
2: (-123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544)
3: (-135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521)
4: (-123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344)
5: (-109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042)
6: (-104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754)
7: (-116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424)
8: (-134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655)
9: (-103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908)

## BASE Result
execution time: IAR + LP analysis = 1.26 + 10.44 = 11.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -195.3645330, upper bound: 195.3645330


# Binary Search by BASE starts (time budget: 2688.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.82974243164062
rel_dist={7: [-195.3643469135574, 195.36434691615057]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.82974243164062
rel_dist={7: [-195.36410562008308, 195.3641056191758]}

## Binary Search Result
Binary search time: 49.27 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2639.04 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3590608, upper bound: 195.3590608
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3590608, upper bound: 195.3590608
time: 9.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.01
Output dim: 7, lower bound: -195.3590608, upper bound: 195.3590608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.01
Output dim: 7, lower bound: -195.3590608, upper bound: 195.3590608

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587577, upper bound: 195.3587592
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587592, upper bound: 195.3587577
time: 9.24 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587577, upper bound: 195.3587592
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587592, upper bound: 195.3587577
time: 9.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 7, lower bound: -195.3587577, upper bound: 195.3587592
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 7, lower bound: -195.3587592, upper bound: 195.3587577
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 7, lower bound: -195.3587577, upper bound: 195.3587592
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 7, lower bound: -195.3587592, upper bound: 195.3587577

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
time: 8.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
time: 9.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
time: 10.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
time: 9.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392936, upper bound: 195.3392947
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.50
Output dim: 7, lower bound: -195.3392947, upper bound: 195.3392936

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
time: 9.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
time: 9.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
time: 9.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
time: 10.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
time: 9.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
time: 10.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
time: 10.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
time: 9.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
time: 9.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
time: 9.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388895, upper bound: 195.3388907
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388918, upper bound: 195.3388878
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388878, upper bound: 195.3388918
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.23
Output dim: 7, lower bound: -195.3388907, upper bound: 195.3388895
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3591544, upper bound: 195.3591544
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3591544, upper bound: 195.3591544
time: 8.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.43
Output dim: 7, lower bound: -195.3591544, upper bound: 195.3591544
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.43
Output dim: 7, lower bound: -195.3591544, upper bound: 195.3591544

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3588459, upper bound: 195.3588532
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3588532, upper bound: 195.3588459
time: 8.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3588459, upper bound: 195.3588532
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3588532, upper bound: 195.3588459
time: 7.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -195.3588459, upper bound: 195.3588532
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -195.3588532, upper bound: 195.3588459
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -195.3588459, upper bound: 195.3588532
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -195.3588532, upper bound: 195.3588459

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393499, upper bound: 195.3393510
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393499, upper bound: 195.3393510
time: 9.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393495, upper bound: 195.3393510
time: 10.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393495, upper bound: 195.3393510
time: 10.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393495
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393495
time: 9.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393499
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393499
time: 11.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393499, upper bound: 195.3393510
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393499, upper bound: 195.3393510
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393495, upper bound: 195.3393510
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393495, upper bound: 195.3393510
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393495
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393495
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393499
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 7, lower bound: -195.3393510, upper bound: 195.3393499

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
time: 8.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
time: 10.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
time: 8.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
time: 11.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
time: 9.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
time: 10.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
time: 9.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
time: 8.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
time: 7.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
time: 8.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
time: 10.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
time: 9.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389537, upper bound: 195.3389563
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389529
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
time: 9.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
time: 10.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
time: 9.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
time: 9.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
time: 9.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
time: 9.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
time: 9.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
time: 9.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
time: 9.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
time: 9.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
time: 9.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
time: 9.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
time: 9.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
time: 9.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120554
time: 11.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120554
time: 9.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120546, upper bound: 195.3120585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120554, upper bound: 195.3120581
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120580, upper bound: 195.3120550
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120543, upper bound: 195.3120581
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120550, upper bound: 195.3120580
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.18
Output dim: 7, lower bound: -195.3120581, upper bound: 195.3120554
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -195.3389529, upper bound: 195.3389563
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -195.3389563, upper bound: 195.3389537
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=207.82974243164062
rel_dist={7: [-195.36447750727785, 195.36447750727785]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3590936, upper bound: 195.3590935
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3590936, upper bound: 195.3590936
time: 11.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.81
Output dim: 7, lower bound: -195.3590936, upper bound: 195.3590935
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.81
Output dim: 7, lower bound: -195.3590936, upper bound: 195.3590936

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587878, upper bound: 195.3587914
time: 9.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587914, upper bound: 195.3587878
time: 9.00 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587878, upper bound: 195.3587914
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3587914, upper bound: 195.3587878
time: 10.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 7, lower bound: -195.3587878, upper bound: 195.3587914
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 7, lower bound: -195.3587914, upper bound: 195.3587878
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 7, lower bound: -195.3587878, upper bound: 195.3587914
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 7, lower bound: -195.3587914, upper bound: 195.3587878

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
time: 8.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
time: 8.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
time: 7.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
time: 8.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393138, upper bound: 195.3393148
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3393148, upper bound: 195.3393138

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389104
time: 9.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
time: 9.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389104
time: 10.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389103, upper bound: 195.3389145
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
time: 9.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389103, upper bound: 195.3389145
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
time: 9.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389103
time: 9.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389103
time: 10.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389104, upper bound: 195.3389145
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
time: 9.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389104, upper bound: 195.3389145
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
time: 8.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389104
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389104
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389103, upper bound: 195.3389145
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389103, upper bound: 195.3389145
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389103
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389120, upper bound: 195.3389137
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389145, upper bound: 195.3389103
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389104, upper bound: 195.3389145
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389104, upper bound: 195.3389145
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.00
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
time: 7.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
time: 8.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
time: 6.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
time: 8.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
time: 9.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
time: 8.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
time: 7.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
time: 9.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
time: 8.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
time: 8.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
time: 8.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
time: 8.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120220
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120220
time: 9.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864
1: -94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567
2: -123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544
3: -135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521
4: -123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344
5: -109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042
6: -104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754
7: -116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424
8: -134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655
9: -103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120211
time: 10.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120211
time: 10.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120211, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120220, upper bound: 195.3120245
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120214
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120241, upper bound: 195.3120209
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120209, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120214, upper bound: 195.3120241
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 7, lower bound: -195.3120245, upper bound: 195.3120211
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 7, lower bound: -195.3389104, upper bound: 195.3389145
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 7, lower bound: -195.3389137, upper bound: 195.3389120
Binary search (step 2): status=Status.UNKNOWN, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=207.82974243164062
rel_dist={7: [-195.3644399904478, 195.36443998460834]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1564.77 seconds
