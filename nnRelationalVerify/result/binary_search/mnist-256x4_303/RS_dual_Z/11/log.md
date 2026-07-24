## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 330.610742861
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250)
1: (-154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879)
2: (-202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273)
3: (-215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854)
4: (-197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931)
5: (-176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317)
6: (-169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464)
7: (-184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930)
8: (-221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588)
9: (-167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152)

## BASE Result
execution time: IAR + LP analysis = 1.19 + 10.39 = 11.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -330.6222845, upper bound: 330.6222845


# Binary Search by BASE starts (time budget: 2688.43 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=333.2668151855469
rel_dist={9: [-330.62178962803773, 330.6217896201217]}

## Binary Search Result
Binary search time: 48.06 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2640.37 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221509, upper bound: 330.6221776
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221509
time: 6.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.88
Output dim: 9, lower bound: -330.6221509, upper bound: 330.6221776
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.88
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221509

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221131, upper bound: 330.6221484
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221131, upper bound: 330.6221484
time: 7.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221131
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221131
time: 8.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 9, lower bound: -330.6221131, upper bound: 330.6221484
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 9, lower bound: -330.6221131, upper bound: 330.6221484
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221131
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.55
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221131

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
time: 8.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
time: 7.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
time: 7.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
time: 8.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
time: 8.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216667, upper bound: 330.6216875
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.93
Output dim: 9, lower bound: -330.6216875, upper bound: 330.6216667

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 8.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
time: 7.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.33
Output dim: 9, lower bound: -330.6177009, upper bound: 330.6177009

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 9.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 10.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 9.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 7.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
time: 8.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084154, upper bound: 330.6084200
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084197, upper bound: 330.6084166
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084170, upper bound: 330.6084177
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084212, upper bound: 330.6084134
time: 9.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084154, upper bound: 330.6084200
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084197, upper bound: 330.6084166
time: 6.56 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084154, upper bound: 330.6084200
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084197, upper bound: 330.6084166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084170, upper bound: 330.6084177
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084212, upper bound: 330.6084134
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084154, upper bound: 330.6084200
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.53
Output dim: 9, lower bound: -330.6084197, upper bound: 330.6084166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127186, upper bound: 330.6127395
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.53
Output dim: 9, lower bound: -330.6127395, upper bound: 330.6127186
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6220002, upper bound: 330.6220156
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6220156, upper bound: 330.6220002
time: 7.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.36
Output dim: 9, lower bound: -330.6220002, upper bound: 330.6220156
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.36
Output dim: 9, lower bound: -330.6220156, upper bound: 330.6220002

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6219634, upper bound: 330.6219826
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6219634, upper bound: 330.6219826
time: 8.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6219826, upper bound: 330.6219634
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6219826, upper bound: 330.6219634
time: 8.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 9, lower bound: -330.6219634, upper bound: 330.6219826
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 9, lower bound: -330.6219634, upper bound: 330.6219826
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 9, lower bound: -330.6219826, upper bound: 330.6219634
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 9, lower bound: -330.6219826, upper bound: 330.6219634

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
time: 9.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
time: 8.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
time: 8.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
time: 8.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215101, upper bound: 330.6215222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.86
Output dim: 9, lower bound: -330.6215222, upper bound: 330.6215101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
time: 8.15 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.25
Output dim: 9, lower bound: -330.6175647, upper bound: 330.6175647

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 8.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 9.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 8.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 8.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 7.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 6.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 9.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
time: 9.26 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.87
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082808, upper bound: 330.6082862
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082865, upper bound: 330.6082804
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082810, upper bound: 330.6082862
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082867, upper bound: 330.6082802
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082808, upper bound: 330.6082862
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082865, upper bound: 330.6082804
time: 7.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082810, upper bound: 330.6082862
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6082867, upper bound: 330.6082802
time: 6.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082808, upper bound: 330.6082862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082865, upper bound: 330.6082804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082810, upper bound: 330.6082862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082867, upper bound: 330.6082802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082808, upper bound: 330.6082862
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082865, upper bound: 330.6082804
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082810, upper bound: 330.6082862
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.79
Output dim: 9, lower bound: -330.6082867, upper bound: 330.6082802
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125853, upper bound: 330.6125907
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.79
Output dim: 9, lower bound: -330.6125907, upper bound: 330.6125853
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217856, upper bound: 330.6217896
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217896, upper bound: 330.6217856
time: 21.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 30.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 30.32
Output dim: 9, lower bound: -330.6217856, upper bound: 330.6217896
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 30.32
Output dim: 9, lower bound: -330.6217896, upper bound: 330.6217856

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217558, upper bound: 330.6217611
time: 11.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217558, upper bound: 330.6217611
time: 10.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217611, upper bound: 330.6217558
time: 10.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6217611, upper bound: 330.6217558
time: 11.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 9, lower bound: -330.6217558, upper bound: 330.6217611
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 9, lower bound: -330.6217558, upper bound: 330.6217611
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 9, lower bound: -330.6217611, upper bound: 330.6217558
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 9, lower bound: -330.6217611, upper bound: 330.6217558

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
time: 10.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
time: 11.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
time: 11.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
time: 12.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
time: 10.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
time: 12.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
time: 11.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213088, upper bound: 330.6213124
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.59
Output dim: 9, lower bound: -330.6213124, upper bound: 330.6213088

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 10.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 12.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 9.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 14.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 14.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 12.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 17.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 11.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 14.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 15.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
time: 10.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.31
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 9.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 10.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 10.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 11.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 17.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 11.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 10.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 10.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 9.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 9.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 10.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 14.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
time: 10.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
time: 10.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250
1: -154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879
2: -202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273
3: -215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854
4: -197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931
5: -176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317
6: -169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464
7: -184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930
8: -221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588
9: -167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123690, upper bound: 330.6123691
time: 13.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123690, upper bound: 330.6123682
time: 10.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123682, upper bound: 330.6123690
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123690, upper bound: 330.6123691
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 9, lower bound: -330.6123690, upper bound: 330.6123682
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 9, lower bound: -330.6174075, upper bound: 330.6174075
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=333.2668151855469
rel_dist={9: [-330.62178962803773, 330.6217896201217]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1824.25 seconds
