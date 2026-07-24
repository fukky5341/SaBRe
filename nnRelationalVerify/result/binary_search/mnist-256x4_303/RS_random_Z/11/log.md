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
execution time: IAR + LP analysis = 1.11 + 10.39 = 11.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -330.6222845, upper bound: 330.6222845


# Binary Search by BASE starts (time budget: 2688.50 seconds, max iter: 100)

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
Binary search time: 48.16 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2640.34 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6209145, upper bound: 330.6209145
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6209145, upper bound: 330.6209145
time: 10.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.90
Output dim: 9, lower bound: -330.6209145, upper bound: 330.6209145
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.90
Output dim: 9, lower bound: -330.6209145, upper bound: 330.6209145

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
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6204312, upper bound: 330.6204312
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6204312, upper bound: 330.6204312
time: 7.50 seconds

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6203429, upper bound: 330.6203429
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6203429, upper bound: 330.6203429
time: 9.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 9, lower bound: -330.6204312, upper bound: 330.6204312
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 9, lower bound: -330.6204312, upper bound: 330.6204312
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 9, lower bound: -330.6203429, upper bound: 330.6203429
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 9, lower bound: -330.6203429, upper bound: 330.6203429

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6017873, upper bound: 330.6017947
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6017873, upper bound: 330.6017947
time: 7.64 seconds

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
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5903503, upper bound: 330.5903521
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5903503, upper bound: 330.5903521
time: 6.53 seconds

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6191411, upper bound: 330.6191616
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6191616, upper bound: 330.6191411
time: 8.04 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5803689, upper bound: 330.5803689
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5803689, upper bound: 330.5803689
time: 6.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.6017873, upper bound: 330.6017947
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.6017873, upper bound: 330.6017947
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.5903503, upper bound: 330.5903521
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.5903503, upper bound: 330.5903521
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.6191411, upper bound: 330.6191616
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.6191616, upper bound: 330.6191411
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.5803689, upper bound: 330.5803689
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.39
Output dim: 9, lower bound: -330.5803689, upper bound: 330.5803689

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6191411, upper bound: 330.6191460
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6191237, upper bound: 330.6191616
time: 8.10 seconds

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
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5843288, upper bound: 330.5843301
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5843288, upper bound: 330.5843301
time: 6.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.48 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.48
Output dim: 9, lower bound: -330.6191411, upper bound: 330.6191460
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.48
Output dim: 9, lower bound: -330.6191237, upper bound: 330.6191616
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.48
Output dim: 9, lower bound: -330.5843288, upper bound: 330.5843301
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.48
Output dim: 9, lower bound: -330.5843288, upper bound: 330.5843301

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6068891, upper bound: 330.6068738
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6068891, upper bound: 330.6068738
time: 8.33 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183154, upper bound: 330.6183666
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183473, upper bound: 330.6183346
time: 7.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.13 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.13
Output dim: 9, lower bound: -330.6068891, upper bound: 330.6068738
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.13
Output dim: 9, lower bound: -330.6068891, upper bound: 330.6068738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 9, lower bound: -330.6183154, upper bound: 330.6183666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.13
Output dim: 9, lower bound: -330.6183473, upper bound: 330.6183346

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6182833, upper bound: 330.6183666
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183154, upper bound: 330.6183246
time: 10.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183458, upper bound: 330.6183346
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183473, upper bound: 330.6183340
time: 7.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.50 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.50
Output dim: 9, lower bound: -330.6182833, upper bound: 330.6183666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.50
Output dim: 9, lower bound: -330.6183154, upper bound: 330.6183246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.50
Output dim: 9, lower bound: -330.6183458, upper bound: 330.6183346
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.50
Output dim: 9, lower bound: -330.6183473, upper bound: 330.6183340

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139248, upper bound: 330.6139801
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139246, upper bound: 330.6139636
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6183005, upper bound: 330.6183126
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6182955, upper bound: 330.6183166
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5708142, upper bound: 330.5708119
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5708142, upper bound: 330.5708119
time: 6.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139555
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139766, upper bound: 330.6139494
time: 7.89 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.15 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6139248, upper bound: 330.6139801
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6139246, upper bound: 330.6139636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6183005, upper bound: 330.6183126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6182955, upper bound: 330.6183166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.5708142, upper bound: 330.5708119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.5708142, upper bound: 330.5708119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139555
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.15
Output dim: 9, lower bound: -330.6139766, upper bound: 330.6139494

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139248, upper bound: 330.6139801
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139234, upper bound: 330.6139798
time: 9.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139246, upper bound: 330.6139636
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139240, upper bound: 330.6139636
time: 8.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6070229, upper bound: 330.6070191
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6070229, upper bound: 330.6070191
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6182856, upper bound: 330.6183121
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6182942, upper bound: 330.6182953
time: 7.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139551
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139555
time: 11.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6076625, upper bound: 330.6076501
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6076625, upper bound: 330.6076501
time: 7.51 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 16.23 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139248, upper bound: 330.6139801
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139234, upper bound: 330.6139798
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139246, upper bound: 330.6139636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139240, upper bound: 330.6139636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6070229, upper bound: 330.6070191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6070229, upper bound: 330.6070191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6182856, upper bound: 330.6183121
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6182942, upper bound: 330.6182953
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139555
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6076625, upper bound: 330.6076501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 16.23
Output dim: 9, lower bound: -330.6076625, upper bound: 330.6076501

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5732042, upper bound: 330.5732065
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5732042, upper bound: 330.5732065
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5732046, upper bound: 330.5732028
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5732046, upper bound: 330.5732028
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
time: 7.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6137770, upper bound: 330.6138106
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6137770, upper bound: 330.6138106
time: 6.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5391997, upper bound: 330.5391946
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5391997, upper bound: 330.5391946
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5807141, upper bound: 330.5807150
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5807141, upper bound: 330.5807150
time: 7.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139575, upper bound: 330.6139551
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139499
time: 7.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5611065, upper bound: 330.5611130
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5611065, upper bound: 330.5611130
time: 6.25 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 13.57 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5732042, upper bound: 330.5732065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5732042, upper bound: 330.5732065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5732046, upper bound: 330.5732028
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5732046, upper bound: 330.5732028
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6137770, upper bound: 330.6138106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6137770, upper bound: 330.6138106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5391997, upper bound: 330.5391946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5391997, upper bound: 330.5391946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5807141, upper bound: 330.5807150
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5807141, upper bound: 330.5807150
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6139575, upper bound: 330.6139551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.6139603, upper bound: 330.6139499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5611065, upper bound: 330.5611130
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 13.57
Output dim: 9, lower bound: -330.5611065, upper bound: 330.5611130

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5299194, upper bound: 330.5299200
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5299194, upper bound: 330.5299200
time: 7.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6107576, upper bound: 330.6108054
time: 7.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5863019, upper bound: 330.5863026
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5863019, upper bound: 330.5863026
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084771, upper bound: 330.6084845
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6084771, upper bound: 330.6084845
time: 8.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5862975, upper bound: 330.5863165
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5862975, upper bound: 330.5863163
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6132910, upper bound: 330.6132666
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6132910, upper bound: 330.6132666
time: 7.15 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 15.64 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5299194, upper bound: 330.5299200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5299194, upper bound: 330.5299200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6107577, upper bound: 330.6108055
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6107576, upper bound: 330.6108054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5863019, upper bound: 330.5863026
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5863019, upper bound: 330.5863026
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6084771, upper bound: 330.6084845
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6084771, upper bound: 330.6084845
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5862975, upper bound: 330.5863165
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.5862975, upper bound: 330.5863163
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6132910, upper bound: 330.6132666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 15.64
Output dim: 9, lower bound: -330.6132910, upper bound: 330.6132666
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6189707, upper bound: 330.6189714
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6189714, upper bound: 330.6189707
time: 6.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.56
Output dim: 9, lower bound: -330.6189707, upper bound: 330.6189714
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.56
Output dim: 9, lower bound: -330.6189714, upper bound: 330.6189707

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6189707, upper bound: 330.6189688
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6189684, upper bound: 330.6189714
time: 6.70 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6071583, upper bound: 330.6071657
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6071583, upper bound: 330.6071657
time: 6.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 9, lower bound: -330.6189707, upper bound: 330.6189688
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 9, lower bound: -330.6189684, upper bound: 330.6189714
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.67
Output dim: 9, lower bound: -330.6071583, upper bound: 330.6071657
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.67
Output dim: 9, lower bound: -330.6071583, upper bound: 330.6071657

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
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109241
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109241
time: 8.39 seconds

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6071861, upper bound: 330.6071908
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6071861, upper bound: 330.6071908
time: 7.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109241
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109241
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.29
Output dim: 9, lower bound: -330.6071861, upper bound: 330.6071908
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.29
Output dim: 9, lower bound: -330.6071861, upper bound: 330.6071908

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109232, upper bound: 330.6109241
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109175
time: 6.92 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108122
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119
time: 11.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.97
Output dim: 9, lower bound: -330.6109232, upper bound: 330.6109241
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.97
Output dim: 9, lower bound: -330.6109269, upper bound: 330.6109175
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.97
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108122
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.97
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5496311, upper bound: 330.5496303
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5496311, upper bound: 330.5496303
time: 5.41 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6099899, upper bound: 330.6099830
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6099899, upper bound: 330.6099830
time: 7.37 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5938718, upper bound: 330.5938720
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5938718, upper bound: 330.5938720
time: 7.22 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119
time: 9.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.5496311, upper bound: 330.5496303
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.5496311, upper bound: 330.5496303
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.6099899, upper bound: 330.6099830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.6099899, upper bound: 330.6099830
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.5938718, upper bound: 330.5938720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.5938718, upper bound: 330.5938720
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 9, lower bound: -330.6108228, upper bound: 330.6108119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5905275, upper bound: 330.5905277
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5905275, upper bound: 330.5905277
time: 7.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6059350, upper bound: 330.6059333
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6059350, upper bound: 330.6059333
time: 8.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.99 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.99
Output dim: 9, lower bound: -330.5905275, upper bound: 330.5905277
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.99
Output dim: 9, lower bound: -330.5905275, upper bound: 330.5905277
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.99
Output dim: 9, lower bound: -330.6059350, upper bound: 330.6059333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.99
Output dim: 9, lower bound: -330.6059350, upper bound: 330.6059333
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6098501, upper bound: 330.6098501
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6098501, upper bound: 330.6098501
time: 6.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.66 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.66
Output dim: 9, lower bound: -330.6098501, upper bound: 330.6098501
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.66
Output dim: 9, lower bound: -330.6098501, upper bound: 330.6098501
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=333.2668151855469
rel_dist={9: [-330.62209603555596, 330.6220960272268]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6076667, upper bound: 330.6076668
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6076667, upper bound: 330.6076667
time: 6.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.76 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.76
Output dim: 9, lower bound: -330.6076667, upper bound: 330.6076668
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.76
Output dim: 9, lower bound: -330.6076667, upper bound: 330.6076667
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=333.2668151855469
rel_dist={9: [-330.62215419250697, 330.62215419250697]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 888.06 seconds
