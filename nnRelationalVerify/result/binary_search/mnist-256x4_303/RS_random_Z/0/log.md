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
execution time: IAR + LP analysis = 1.26 + 10.65 = 11.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -195.3645330, upper bound: 195.3645330


# Binary Search by BASE starts (time budget: 2688.09 seconds, max iter: 100)

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
Binary search time: 49.17 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2638.92 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3644208, upper bound: 195.3644193
time: 10.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3644193, upper bound: 195.3644208
time: 9.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.21
Output dim: 7, lower bound: -195.3644208, upper bound: 195.3644193
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.21
Output dim: 7, lower bound: -195.3644193, upper bound: 195.3644208

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2949482, upper bound: 195.2949679
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2949482, upper bound: 195.2949679
time: 7.71 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3643510, upper bound: 195.3643456
time: 9.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.98 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 19.98
Output dim: 7, lower bound: -195.2949482, upper bound: 195.2949679
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 19.98
Output dim: 7, lower bound: -195.2949482, upper bound: 195.2949679
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 7, lower bound: -195.3643510, upper bound: 195.3643456

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
time: 9.51 seconds

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3385554, upper bound: 195.3385557
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3385554, upper bound: 195.3385557
time: 9.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.27 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.27
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.27
Output dim: 7, lower bound: -195.3643442, upper bound: 195.3643519
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.27
Output dim: 7, lower bound: -195.3385554, upper bound: 195.3385557
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.27
Output dim: 7, lower bound: -195.3385554, upper bound: 195.3385557

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3638978, upper bound: 195.3639234
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3639005, upper bound: 195.3639133
time: 10.04 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3192810, upper bound: 195.3192747
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3192810, upper bound: 195.3192747
time: 8.19 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.77 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.77
Output dim: 7, lower bound: -195.3638978, upper bound: 195.3639234
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.77
Output dim: 7, lower bound: -195.3639005, upper bound: 195.3639133
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.77
Output dim: 7, lower bound: -195.3192810, upper bound: 195.3192747
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.77
Output dim: 7, lower bound: -195.3192810, upper bound: 195.3192747

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3494786, upper bound: 195.3494735
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3494786, upper bound: 195.3494734
time: 8.73 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3611489, upper bound: 195.3611548
time: 10.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3611489, upper bound: 195.3611548
time: 10.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.70 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.70
Output dim: 7, lower bound: -195.3494786, upper bound: 195.3494735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.70
Output dim: 7, lower bound: -195.3494786, upper bound: 195.3494734
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.70
Output dim: 7, lower bound: -195.3611489, upper bound: 195.3611548
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.70
Output dim: 7, lower bound: -195.3611489, upper bound: 195.3611548

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2582124, upper bound: 195.2582271
time: 9.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2582124, upper bound: 195.2582271
time: 9.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3274698, upper bound: 195.3274965
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3274698, upper bound: 195.3274965
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2759924, upper bound: 195.2759844
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2759924, upper bound: 195.2759844
time: 9.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3564842, upper bound: 195.3564763
time: 9.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3564842, upper bound: 195.3564763
time: 10.13 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.88 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.2582124, upper bound: 195.2582271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.2582124, upper bound: 195.2582271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.3274698, upper bound: 195.3274965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.3274698, upper bound: 195.3274965
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.2759924, upper bound: 195.2759844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.2759924, upper bound: 195.2759844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.3564842, upper bound: 195.3564763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.88
Output dim: 7, lower bound: -195.3564842, upper bound: 195.3564763

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3531883, upper bound: 195.3531866
time: 9.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3531883, upper bound: 195.3531866
time: 11.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3082351, upper bound: 195.3082442
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3082351, upper bound: 195.3082442
time: 9.76 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 20.35 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 20.35
Output dim: 7, lower bound: -195.3531883, upper bound: 195.3531866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.35
Output dim: 7, lower bound: -195.3531883, upper bound: 195.3531866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.35
Output dim: 7, lower bound: -195.3082351, upper bound: 195.3082442
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 20.35
Output dim: 7, lower bound: -195.3082351, upper bound: 195.3082442

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453534, upper bound: 195.3453663
time: 11.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664
time: 8.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3232085, upper bound: 195.3232211
time: 10.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3232085, upper bound: 195.3232211
time: 9.91 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 21.91 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 21.91
Output dim: 7, lower bound: -195.3453534, upper bound: 195.3453663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 21.91
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 21.91
Output dim: 7, lower bound: -195.3232085, upper bound: 195.3232211
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.91
Output dim: 7, lower bound: -195.3232085, upper bound: 195.3232211

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3307128, upper bound: 195.3307380
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3307128, upper bound: 195.3307380
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664
time: 10.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664
time: 10.37 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 21.91 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 21.91
Output dim: 7, lower bound: -195.3307128, upper bound: 195.3307380
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 21.91
Output dim: 7, lower bound: -195.3307128, upper bound: 195.3307380
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 21.91
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 21.91
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453664

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2561586, upper bound: 195.2561667
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2561586, upper bound: 195.2561667
time: 7.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453517, upper bound: 195.3453664
time: 12.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453648
time: 11.01 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 24.67 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 24.67
Output dim: 7, lower bound: -195.2561586, upper bound: 195.2561667
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 24.67
Output dim: 7, lower bound: -195.2561586, upper bound: 195.2561667
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 24.67
Output dim: 7, lower bound: -195.3453517, upper bound: 195.3453664
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 24.67
Output dim: 7, lower bound: -195.3453532, upper bound: 195.3453648

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3284947, upper bound: 195.3285278
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3284947, upper bound: 195.3285278
time: 8.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2863749, upper bound: 195.2863107
time: 8.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2863749, upper bound: 195.2863107
time: 8.53 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 18.32 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 18.32
Output dim: 7, lower bound: -195.3284947, upper bound: 195.3285278
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 18.32
Output dim: 7, lower bound: -195.3284947, upper bound: 195.3285278
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 18.32
Output dim: 7, lower bound: -195.2863749, upper bound: 195.2863107
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 18.32
Output dim: 7, lower bound: -195.2863749, upper bound: 195.2863107
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2954111, upper bound: 195.2954111
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2954111, upper bound: 195.2954111
time: 8.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.60 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 16.60
Output dim: 7, lower bound: -195.2954111, upper bound: 195.2954111
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 16.60
Output dim: 7, lower bound: -195.2954111, upper bound: 195.2954111
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=207.82974243164062
rel_dist={7: [-195.36447750727785, 195.36447750727785]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3617263, upper bound: 195.3617263
time: 10.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3617263, upper bound: 195.3617263
time: 10.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.54
Output dim: 7, lower bound: -195.3617263, upper bound: 195.3617263
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.54
Output dim: 7, lower bound: -195.3617263, upper bound: 195.3617263

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
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3558344, upper bound: 195.3558344
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3558344, upper bound: 195.3558344
time: 9.25 seconds

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
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2453459, upper bound: 195.2453432
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2453459, upper bound: 195.2453432
time: 6.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.08
Output dim: 7, lower bound: -195.3558344, upper bound: 195.3558344
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.08
Output dim: 7, lower bound: -195.3558344, upper bound: 195.3558344
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.08
Output dim: 7, lower bound: -195.2453459, upper bound: 195.2453432
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.08
Output dim: 7, lower bound: -195.2453459, upper bound: 195.2453432

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2561802, upper bound: 195.2561802
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2561802, upper bound: 195.2561802
time: 6.15 seconds

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3092720, upper bound: 195.3092720
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3092720, upper bound: 195.3092720
time: 8.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.2561802, upper bound: 195.2561802
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.2561802, upper bound: 195.2561802
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3092720, upper bound: 195.3092720
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.77
Output dim: 7, lower bound: -195.3092720, upper bound: 195.3092720
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=207.82974243164062
rel_dist={7: [-195.3645145306792, 195.36451453067923]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2954403, upper bound: 195.2954403
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2954403, upper bound: 195.2954403
time: 7.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.00 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 14.00
Output dim: 7, lower bound: -195.2954403, upper bound: 195.2954403
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 14.00
Output dim: 7, lower bound: -195.2954403, upper bound: 195.2954403
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=207.82974243164062
rel_dist={7: [-195.36453304237992, 195.3645330423799]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 629.53 seconds
