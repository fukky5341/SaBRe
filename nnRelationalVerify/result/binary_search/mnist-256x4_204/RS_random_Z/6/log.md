## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 6.28359642836
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192)
1: (-2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467)
2: (-3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980)
3: (-4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634)
4: (-5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445)
5: (-4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435)
6: (-4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641)
7: (-3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932)
8: (-5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192)
9: (-3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559)

## BASE Result
execution time: IAR + LP analysis = 1.27 + 4.02 = 5.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -7.1077352, upper bound: 7.1077352


# Binary Search by BASE starts (time budget: 2694.72 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search Result
Binary search time: 23.03 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2671.68 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9119397
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9119397
time: 1.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.37
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9119397
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.37
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9119397

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9118449
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9118449, upper bound: 6.9119397
time: 1.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8127653
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8127653
time: 1.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.19
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9118449
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.19
Output dim: 6, lower bound: -6.9118449, upper bound: 6.9119397
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.19
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8127653
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.19
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8127653

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9119369, upper bound: 6.9118449
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9118445
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9117785, upper bound: 6.9119397
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9118449, upper bound: 6.9118920
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126816, upper bound: 6.8127653
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8126816
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126816, upper bound: 6.8127653
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8126816
time: 1.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.9119369, upper bound: 6.9118449
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.9119397, upper bound: 6.9118445
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.9117785, upper bound: 6.9119397
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.9118449, upper bound: 6.9118920
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.8126816, upper bound: 6.8127653
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8126816
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.8126816, upper bound: 6.8127653
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 6, lower bound: -6.8127653, upper bound: 6.8126816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8649671, upper bound: 6.8648412
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8649671, upper bound: 6.8648408
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8715844, upper bound: 6.8715441
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8716281, upper bound: 6.8715154
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9117785, upper bound: 6.9119397
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9117778, upper bound: 6.9119369
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126838, upper bound: 6.8126816
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126838, upper bound: 6.8126816
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130131
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130131
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7711482, upper bound: 6.7710983
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7711777, upper bound: 6.7710722
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081760
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080933, upper bound: 6.8081442
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600878
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600878
time: 1.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8649671, upper bound: 6.8648412
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8649671, upper bound: 6.8648408
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8715844, upper bound: 6.8715441
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8716281, upper bound: 6.8715154
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.9117785, upper bound: 6.9119397
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.9117778, upper bound: 6.9119369
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8126838, upper bound: 6.8126816
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8126838, upper bound: 6.8126816
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.7711482, upper bound: 6.7710983
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.7711777, upper bound: 6.7710722
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081760
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.8080933, upper bound: 6.8081442
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600878
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.93
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600878

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2750561, upper bound: 6.2750575
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2750561, upper bound: 6.2750575
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8450382, upper bound: 6.8449555
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8450585, upper bound: 6.8449212
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8523561, upper bound: 6.8523277
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8523561, upper bound: 6.8523277
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291121
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291121
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5719192, upper bound: 6.5721491
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5719192, upper bound: 6.5721491
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8923525, upper bound: 6.8925430
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8923962, upper bound: 6.8925057
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126653, upper bound: 6.8126816
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126835, upper bound: 6.8126812
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710755, upper bound: 6.7710983
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130010
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130010
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7191887
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7191887
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081447
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080606, upper bound: 6.8081760
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559484
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559484
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7559484, upper bound: 6.7559475
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7560000, upper bound: 6.7558917
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600857
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601392, upper bound: 6.7600878
time: 2.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.2750561, upper bound: 6.2750575
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.2750561, upper bound: 6.2750575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8450382, upper bound: 6.8449555
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8450585, upper bound: 6.8449212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8523561, upper bound: 6.8523277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8523561, upper bound: 6.8523277
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291121
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.5719192, upper bound: 6.5721491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.5719192, upper bound: 6.5721491
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8923525, upper bound: 6.8925430
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8923962, upper bound: 6.8925057
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8126653, upper bound: 6.8126816
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8126835, upper bound: 6.8126812
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7710755, upper bound: 6.7710983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130010
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6130010
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7191887
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7191887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.8080606, upper bound: 6.8081760
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559484
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559484
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7559484, upper bound: 6.7559475
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7560000, upper bound: 6.7558917
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7601403, upper bound: 6.7600857
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 6, lower bound: -6.7601392, upper bound: 6.7600878

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7828519, upper bound: 6.7828246
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7828519, upper bound: 6.7828246
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7959859, upper bound: 6.7958339
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7959859, upper bound: 6.7958339
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8523235, upper bound: 6.8522126
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8522604, upper bound: 6.8522951
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8475236, upper bound: 6.8474933
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8475228, upper bound: 6.8474921
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291101
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292599, upper bound: 6.5291121
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5103993, upper bound: 6.5102579
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5103993, upper bound: 6.5102142
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5672243, upper bound: 6.5674562
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5672331, upper bound: 6.5674292
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5718277, upper bound: 6.5720143
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5718277, upper bound: 6.5720143
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7243687, upper bound: 6.7244895
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7243687, upper bound: 6.7244895
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8923962, upper bound: 6.8924732
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8923665, upper bound: 6.8925057
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7963393, upper bound: 6.7963463
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7963341, upper bound: 6.7963489
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126768, upper bound: 6.8126754
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8126835, upper bound: 6.8126812
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7264092
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7264092
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
time: 2.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5768147, upper bound: 6.5767984
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5768143, upper bound: 6.5767951
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129962, upper bound: 6.6130124
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6129489
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129449, upper bound: 6.6130010
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130088
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129818, upper bound: 6.6130131
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7189644
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7189760, upper bound: 6.7191887
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5338479, upper bound: 6.5338283
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5338479, upper bound: 6.5338283
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545321, upper bound: 6.7544522
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545316, upper bound: 6.7544522
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081445
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8080638, upper bound: 6.8081447
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7906906, upper bound: 6.7908125
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7906930, upper bound: 6.7908124
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7378444, upper bound: 6.7378448
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7378471, upper bound: 6.7378448
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559355
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7559453, upper bound: 6.7559484
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7398859, upper bound: 6.7398896
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7398917, upper bound: 6.7398891
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4433766, upper bound: 6.4433264
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4433766, upper bound: 6.4433264
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7420638, upper bound: 6.7419692
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7420637, upper bound: 6.7419707
time: 2.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601392, upper bound: 6.7600714
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7601311, upper bound: 6.7600878
time: 1.70 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7828519, upper bound: 6.7828246
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7828519, upper bound: 6.7828246
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7959859, upper bound: 6.7958339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7959859, upper bound: 6.7958339
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8523235, upper bound: 6.8522126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8522604, upper bound: 6.8522951
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8475236, upper bound: 6.8474933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8475228, upper bound: 6.8474921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5292599, upper bound: 6.5291121
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5103993, upper bound: 6.5102579
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5103993, upper bound: 6.5102142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5672243, upper bound: 6.5674562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5672331, upper bound: 6.5674292
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5718277, upper bound: 6.5720143
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5718277, upper bound: 6.5720143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7243687, upper bound: 6.7244895
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7243687, upper bound: 6.7244895
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8923962, upper bound: 6.8924732
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8923665, upper bound: 6.8925057
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7963393, upper bound: 6.7963463
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7963341, upper bound: 6.7963489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8126768, upper bound: 6.8126754
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8126835, upper bound: 6.8126812
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7264092
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7264092
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5768147, upper bound: 6.5767984
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5768143, upper bound: 6.5767951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6129962, upper bound: 6.6130124
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6129489
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6129449, upper bound: 6.6130010
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130088
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.6129818, upper bound: 6.6130131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7191257, upper bound: 6.7189644
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7189760, upper bound: 6.7191887
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5338479, upper bound: 6.5338283
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.5338479, upper bound: 6.5338283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7545321, upper bound: 6.7544522
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7545316, upper bound: 6.7544522
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7545342, upper bound: 6.7544576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8080642, upper bound: 6.8081445
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.8080638, upper bound: 6.8081447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7906906, upper bound: 6.7908125
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7906930, upper bound: 6.7908124
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7378444, upper bound: 6.7378448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7378471, upper bound: 6.7378448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7559475, upper bound: 6.7559355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7559453, upper bound: 6.7559484
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7398859, upper bound: 6.7398896
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7398917, upper bound: 6.7398891
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.4433766, upper bound: 6.4433264
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.4433766, upper bound: 6.4433264
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7420638, upper bound: 6.7419692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7420637, upper bound: 6.7419707
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7601392, upper bound: 6.7600714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 6, lower bound: -6.7601311, upper bound: 6.7600878

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6807390, upper bound: 6.6807181
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6807390, upper bound: 6.6807181
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7828519, upper bound: 6.7827177
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7827265, upper bound: 6.7828246
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7959859, upper bound: 6.7957915
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7959722, upper bound: 6.7958339
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7957633, upper bound: 6.7956195
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7957633, upper bound: 6.7956195
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8474906, upper bound: 6.8473800
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8474898, upper bound: 6.8473713
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8522600, upper bound: 6.8522923
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8522420, upper bound: 6.8522948
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2231256, upper bound: 6.2231067
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2231256, upper bound: 6.2231067
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8362271, upper bound: 6.8361619
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8362271, upper bound: 6.8361619
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292639, upper bound: 6.5291027
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5292474, upper bound: 6.5291101
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5103992, upper bound: 6.5102579
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5103991, upper bound: 6.5102142
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5013828, upper bound: 6.5012558
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5013828, upper bound: 6.5012558
time: 2.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5044147, upper bound: 6.5042701
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5044147, upper bound: 6.5042701
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5672243, upper bound: 6.5674473
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5672181, upper bound: 6.5674562
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3567748, upper bound: 6.3567912
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3567748, upper bound: 6.3567912
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4668417, upper bound: 6.4668694
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4668417, upper bound: 6.4668694
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5718226, upper bound: 6.5720039
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5718277, upper bound: 6.5720143
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6903379, upper bound: 6.6905006
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6903379, upper bound: 6.6905006
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5929102, upper bound: 6.5929670
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5929102, upper bound: 6.5929670
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8519439, upper bound: 6.8520669
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8519792, upper bound: 6.8520313
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8874611, upper bound: 6.8876025
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8874653, upper bound: 6.8876014
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7303524, upper bound: 6.7302561
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7303524, upper bound: 6.7302561
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7961219, upper bound: 6.7961570
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7961219, upper bound: 6.7961570
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7963463, upper bound: 6.7963404
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7963430, upper bound: 6.7963447
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129671, upper bound: 6.6130050
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129671, upper bound: 6.6130050
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7264076
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263455, upper bound: 6.7264092
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263487, upper bound: 6.7263642
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7263314, upper bound: 6.7264092
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7190314, upper bound: 6.7190579
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7190314, upper bound: 6.7190579
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710935, upper bound: 6.7710718
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710855, upper bound: 6.7710722
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5596693, upper bound: 6.5596372
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5596693, upper bound: 6.5596372
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5565022, upper bound: 6.5564680
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5565018, upper bound: 6.5564682
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5720291, upper bound: 6.5720605
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5720291, upper bound: 6.5720605
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5913676, upper bound: 6.5914109
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5913676, upper bound: 6.5914109
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6130052, upper bound: 6.6129371
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129812, upper bound: 6.6129489
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129449, upper bound: 6.6129920
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129380, upper bound: 6.6130010
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129962, upper bound: 6.6130061
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129963, upper bound: 6.6130088
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129818, upper bound: 6.6129988
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6129513, upper bound: 6.6130131
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6926425, upper bound: 6.6924246
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6926136, upper bound: 6.6924365
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7189760, upper bound: 6.7191159
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7189579, upper bound: 6.7191887
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5338479, upper bound: 6.5338021
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5338416, upper bound: 6.5338283
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
time: 3.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0877881, upper bound: 7.0877879
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0877881, upper bound: 7.0877879
time: 2.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071455, upper bound: 7.1071524
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071455
time: 4.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 6, lower bound: -7.0877881, upper bound: 7.0877879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 6, lower bound: -7.0877881, upper bound: 7.0877879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 6, lower bound: -7.1071455, upper bound: 7.1071524
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071455

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9917522
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9917522
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6339407, upper bound: 6.6339222
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6339407, upper bound: 6.6339222
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790809
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790809
time: 2.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071364
time: 2.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071455, upper bound: 7.1071455
time: 2.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9917522
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9917522
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.6339407, upper bound: 6.6339222
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.6339407, upper bound: 6.6339222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790809
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790809
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071364
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 6, lower bound: -7.1071455, upper bound: 7.1071455

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6630613, upper bound: 6.6630613
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6630613, upper bound: 6.6630613
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9916951
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9916984, upper bound: 6.9917522
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6272333
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6272333
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6339284, upper bound: 6.6339148
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6339284, upper bound: 6.6339148
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790789
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9790212, upper bound: 6.9790809
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961151
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961151
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071313, upper bound: 7.1071177
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071313, upper bound: 7.1071177
time: 2.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019907
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019907
time: 3.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6630613, upper bound: 6.6630613
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6630613, upper bound: 6.6630613
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.9917513, upper bound: 6.9916951
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.9916984, upper bound: 6.9917522
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6272333
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6272333
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6339284, upper bound: 6.6339148
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.6339284, upper bound: 6.6339148
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.9790215, upper bound: 6.9790789
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.9790212, upper bound: 6.9790809
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961151
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961151
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -7.1071313, upper bound: 7.1071177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -7.1071313, upper bound: 7.1071177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019907
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.19
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019907

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458399
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458399
time: 2.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665957
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665957
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5805396, upper bound: 6.5805448
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5805396, upper bound: 6.5805448
time: 2.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442978, upper bound: 6.9443334
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442978, upper bound: 6.9443334
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4766108
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4766108
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6271765
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6272057, upper bound: 6.6272333
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5928321, upper bound: 6.5928342
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5928439, upper bound: 6.5928200
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2136839, upper bound: 6.2136683
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2136839, upper bound: 6.2136683
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9536944, upper bound: 6.9537340
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9536896, upper bound: 6.9537439
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556294
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556294
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961103
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960845, upper bound: 6.7961151
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961103
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960845, upper bound: 6.7961151
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1068801, upper bound: 7.1068516
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1068799, upper bound: 7.1068517
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071089, upper bound: 7.1070513
time: 77.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1070643, upper bound: 7.1070823
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019572
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1019583, upper bound: 7.1019907
time: 3.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1004441, upper bound: 7.1004563
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1004485, upper bound: 7.1004516
time: 2.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458399
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458399
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665957
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.5805396, upper bound: 6.5805448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.5805396, upper bound: 6.5805448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.9442978, upper bound: 6.9443334
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.9442978, upper bound: 6.9443334
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4766108
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4766108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.6272500, upper bound: 6.6271765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.6272057, upper bound: 6.6272333
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.5928321, upper bound: 6.5928342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.5928439, upper bound: 6.5928200
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.2136839, upper bound: 6.2136683
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.2136839, upper bound: 6.2136683
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.9536944, upper bound: 6.9537340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.9536896, upper bound: 6.9537439
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.7960845, upper bound: 6.7961151
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.7960851, upper bound: 6.7961103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -6.7960845, upper bound: 6.7961151
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1068801, upper bound: 7.1068516
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1068799, upper bound: 7.1068517
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1071089, upper bound: 7.1070513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1070643, upper bound: 7.1070823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1019989, upper bound: 7.1019572
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1019583, upper bound: 7.1019907
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1004441, upper bound: 7.1004563
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 6, lower bound: -7.1004485, upper bound: 7.1004516

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458228
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6458268, upper bound: 6.6458399
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6198522, upper bound: 6.6198494
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6198494, upper bound: 6.6198504
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665732
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665733, upper bound: 6.3665957
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665808, upper bound: 6.3665957
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665781
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5365350, upper bound: 6.5365469
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5365351, upper bound: 6.5365434
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193942
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193942
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591262
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591262
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4545201, upper bound: 6.4545215
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4545196, upper bound: 6.4545221
time: 2.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765992
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4765977, upper bound: 6.4766108
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765875
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765875
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2136686, upper bound: 6.2136827
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2136686, upper bound: 6.2136827
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5887281, upper bound: 6.5887292
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5887284, upper bound: 6.5887241
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5928439, upper bound: 6.5928015
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5928300, upper bound: 6.5928200
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9536944, upper bound: 6.9536454
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9536344, upper bound: 6.9537340
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556209
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556185, upper bound: 6.3556294
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556113, upper bound: 6.3556294
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556256
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7918179, upper bound: 6.7918541
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7918190, upper bound: 6.7918458
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960753, upper bound: 6.7960783
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7960612, upper bound: 6.7961005
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
time: 3.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
time: 2.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
time: 2.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071089, upper bound: 7.1070456
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071069, upper bound: 7.1070513
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1019280, upper bound: 7.1018163
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1018516, upper bound: 7.1018849
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1018846, upper bound: 7.1018432
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1018308, upper bound: 7.1019207
time: 2.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
time: 2.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235
time: 1.96 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6458420, upper bound: 6.6458228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6458268, upper bound: 6.6458399
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6198522, upper bound: 6.6198494
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6198494, upper bound: 6.6198504
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3665733, upper bound: 6.3665957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3665808, upper bound: 6.3665957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5365350, upper bound: 6.5365469
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5365351, upper bound: 6.5365434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193942
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591262
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4545201, upper bound: 6.4545215
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4545196, upper bound: 6.4545221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4765977, upper bound: 6.4766108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.4766109, upper bound: 6.4765875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.2136686, upper bound: 6.2136827
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.2136686, upper bound: 6.2136827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5887281, upper bound: 6.5887292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5887284, upper bound: 6.5887241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5928439, upper bound: 6.5928015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.5928300, upper bound: 6.5928200
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9536944, upper bound: 6.9536454
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9536344, upper bound: 6.9537340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556209
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3556185, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3556113, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556256
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7918179, upper bound: 6.7918541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7918190, upper bound: 6.7918458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7960753, upper bound: 6.7960783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7960612, upper bound: 6.7961005
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1071089, upper bound: 7.1070456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1071069, upper bound: 7.1070513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1019280, upper bound: 7.1018163
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1018516, upper bound: 7.1018849
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1018846, upper bound: 7.1018432
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.1018308, upper bound: 7.1019207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6320765, upper bound: 6.6320464
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6320758, upper bound: 6.6320486
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6457879, upper bound: 6.6458399
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6458268, upper bound: 6.6457972
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6198522, upper bound: 6.6198066
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6198128, upper bound: 6.6198494
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4984361, upper bound: 6.4984563
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4984361, upper bound: 6.4984563
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665707
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665809, upper bound: 6.3665732
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2693823, upper bound: 6.2694250
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2693823, upper bound: 6.2694250
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3524278, upper bound: 6.3524526
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3524278, upper bound: 6.3524525
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665780
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3665918, upper bound: 6.3665781
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3894825, upper bound: 6.3894803
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3894825, upper bound: 6.3894803
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5365205, upper bound: 6.5365434
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5365351, upper bound: 6.5365115
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193580
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3193082, upper bound: 6.3193942
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3066929, upper bound: 6.3067251
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3067024, upper bound: 6.3067083
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9040904, upper bound: 6.9041292
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9041022, upper bound: 6.9041165
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591154
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8590968, upper bound: 6.8591262
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591259
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8591036, upper bound: 6.8591262
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4394734, upper bound: 6.4394713
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4394705, upper bound: 6.4394746
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4545196, upper bound: 6.4544997
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4544994, upper bound: 6.4545221
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4412130, upper bound: 6.4411944
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4412084, upper bound: 6.4412014
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4411954, upper bound: 6.4412102
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4411930, upper bound: 6.4412166
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4573671, upper bound: 6.4573570
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4573670, upper bound: 6.4573601
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4728821, upper bound: 6.4728876
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4729077, upper bound: 6.4728364
time: 1.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6320765, upper bound: 6.6320464
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6320758, upper bound: 6.6320486
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6457879, upper bound: 6.6458399
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6458268, upper bound: 6.6457972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6198522, upper bound: 6.6198066
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.6198128, upper bound: 6.6198494
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4984361, upper bound: 6.4984563
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4984361, upper bound: 6.4984563
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665707
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3665809, upper bound: 6.3665732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.2693823, upper bound: 6.2694250
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.2693823, upper bound: 6.2694250
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3524278, upper bound: 6.3524526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3524278, upper bound: 6.3524525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3665952, upper bound: 6.3665780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3665918, upper bound: 6.3665781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3894825, upper bound: 6.3894803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3894825, upper bound: 6.3894803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.5365205, upper bound: 6.5365434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.5365351, upper bound: 6.5365115
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3193429, upper bound: 6.3193580
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3193082, upper bound: 6.3193942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3066929, upper bound: 6.3067251
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.3067024, upper bound: 6.3067083
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.9442741, upper bound: 6.9443054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.9040904, upper bound: 6.9041292
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.9041022, upper bound: 6.9041165
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591154
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.8590968, upper bound: 6.8591262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.8591044, upper bound: 6.8591259
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.8591036, upper bound: 6.8591262
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4394734, upper bound: 6.4394713
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4394705, upper bound: 6.4394746
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4545196, upper bound: 6.4544997
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4544994, upper bound: 6.4545221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4412130, upper bound: 6.4411944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4412084, upper bound: 6.4412014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4411954, upper bound: 6.4412102
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4411930, upper bound: 6.4412166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4573671, upper bound: 6.4573570
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4573670, upper bound: 6.4573601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4728821, upper bound: 6.4728876
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 6, lower bound: -6.4729077, upper bound: 6.4728364
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.5887281, upper bound: 6.5887292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.5887284, upper bound: 6.5887241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.5928439, upper bound: 6.5928015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.5928300, upper bound: 6.5928200
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.9536944, upper bound: 6.9536454
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.9536344, upper bound: 6.9537340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.9004670, upper bound: 6.9005144
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556209
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.3556185, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.3556113, upper bound: 6.3556294
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.3556193, upper bound: 6.3556256
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.6945766, upper bound: 6.6946126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7918179, upper bound: 6.7918541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7918190, upper bound: 6.7918458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7960753, upper bound: 6.7960783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7960612, upper bound: 6.7961005
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.7696178, upper bound: 6.7696658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0929073, upper bound: 7.0928649
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0997235, upper bound: 7.0997047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1071089, upper bound: 7.1070456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1071069, upper bound: 7.1070513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1026109, upper bound: 7.1026351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1019280, upper bound: 7.1018163
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1018516, upper bound: 7.1018849
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1018846, upper bound: 7.1018432
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.1018308, upper bound: 7.1019207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -7.0987721, upper bound: 7.0987949
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 6, lower bound: -6.6450208, upper bound: 6.6450235
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0551900, upper bound: 7.0551900
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0551900, upper bound: 7.0551900
time: 2.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.43
Output dim: 6, lower bound: -7.0551900, upper bound: 7.0551900
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.43
Output dim: 6, lower bound: -7.0551900, upper bound: 7.0551900

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9481377, upper bound: 6.9481377
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9481377, upper bound: 6.9481377
time: 2.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321482
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321482
time: 2.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -6.9481377, upper bound: 6.9481377
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -6.9481377, upper bound: 6.9481377
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321482
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321482

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696467
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696467
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8808900, upper bound: 6.8808900
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8808900, upper bound: 6.8808900
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040176
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040176
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321481
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321481, upper bound: 6.8321482
time: 2.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696467
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696467
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.8808900, upper bound: 6.8808900
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.8808900, upper bound: 6.8808900
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040176
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040176
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321481
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.34
Output dim: 6, lower bound: -6.8321481, upper bound: 6.8321482

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696466
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696466, upper bound: 6.6696467
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.1746264, upper bound: 6.1746264
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.1746264, upper bound: 6.1746264
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554552, upper bound: 6.8554550
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554552
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231566
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231566
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6040156, upper bound: 6.6040176
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040156
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011914, upper bound: 6.6011965
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011965, upper bound: 6.6011914
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321460
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321465, upper bound: 6.8321481
time: 13.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321302, upper bound: 6.8320969
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8320960, upper bound: 6.8321303
time: 4.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6696467, upper bound: 6.6696466
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6696466, upper bound: 6.6696467
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.1746264, upper bound: 6.1746264
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.1746264, upper bound: 6.1746264
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8554552, upper bound: 6.8554550
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554552
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231566
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231566
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6040156, upper bound: 6.6040176
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6040176, upper bound: 6.6040156
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6011914, upper bound: 6.6011965
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.6011965, upper bound: 6.6011914
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8321482, upper bound: 6.8321460
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8321465, upper bound: 6.8321481
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8321302, upper bound: 6.8320969
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.79
Output dim: 6, lower bound: -6.8320960, upper bound: 6.8321303

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696455, upper bound: 6.6696326
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696323, upper bound: 6.6696454
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550229, upper bound: 6.6550262
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550262, upper bound: 6.6550232
time: 2.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554552, upper bound: 6.8554455
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554458, upper bound: 6.8554550
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554353
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554353, upper bound: 6.8554552
time: 2.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7987080
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7987080, upper bound: 6.7987079
time: 2.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231514
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8231514, upper bound: 6.8231566
time: 2.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081630
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081630
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918171
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918171
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011914, upper bound: 6.6011926
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011906, upper bound: 6.6011965
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011916, upper bound: 6.6011914
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011965, upper bound: 6.6011903
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8284476, upper bound: 6.8284454
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8284481, upper bound: 6.8284424
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635016
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635016
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321298, upper bound: 6.8320969
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321302, upper bound: 6.8320960
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8185975, upper bound: 6.8186471
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8186138, upper bound: 6.8186359
time: 2.52 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6696455, upper bound: 6.6696326
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6696323, upper bound: 6.6696454
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6550229, upper bound: 6.6550262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6550262, upper bound: 6.6550232
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8554552, upper bound: 6.8554455
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8554458, upper bound: 6.8554550
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554353
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8554353, upper bound: 6.8554552
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7987080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.7987080, upper bound: 6.7987079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8231566, upper bound: 6.8231514
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8231514, upper bound: 6.8231566
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918171
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6011914, upper bound: 6.6011926
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6011906, upper bound: 6.6011965
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6011916, upper bound: 6.6011914
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6011965, upper bound: 6.6011903
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8284476, upper bound: 6.8284454
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8284481, upper bound: 6.8284424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8321298, upper bound: 6.8320969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8321302, upper bound: 6.8320960
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8185975, upper bound: 6.8186471
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.87
Output dim: 6, lower bound: -6.8186138, upper bound: 6.8186359

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696455, upper bound: 6.6696260
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6696414, upper bound: 6.6696326
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6527381, upper bound: 6.6527510
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6527380, upper bound: 6.6527510
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550229, upper bound: 6.6550262
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550228, upper bound: 6.6550253
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6315562, upper bound: 6.6315501
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6315534, upper bound: 6.6315544
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7986861
time: 13.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7986861
time: 12.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201689, upper bound: 6.8201809
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201733, upper bound: 6.8201759
time: 2.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554242
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554456, upper bound: 6.8554353
time: 2.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087561, upper bound: 6.7087601
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087561, upper bound: 6.7087601
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489018
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489018
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7539886, upper bound: 6.7539809
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7539886, upper bound: 6.7539809
time: 2.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4790513, upper bound: 6.4790513
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4790513, upper bound: 6.4790513
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3923435, upper bound: 6.3923464
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3923465, upper bound: 6.3923435
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4081632, upper bound: 6.4081630
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081624
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4742590, upper bound: 6.4742643
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4742642, upper bound: 6.4742604
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4918163, upper bound: 6.4918171
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918163
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5705275, upper bound: 6.5705298
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5705275, upper bound: 6.5705298
time: 2.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011903, upper bound: 6.6011965
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011906, upper bound: 6.6011916
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011916, upper bound: 6.6011902
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011884, upper bound: 6.6011914
time: 2.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5705400, upper bound: 6.5705263
time: 2.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5705400, upper bound: 6.5705263
time: 2.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011902, upper bound: 6.6011926
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6011902, upper bound: 6.6011926
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8284469, upper bound: 6.8284423
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8284482, upper bound: 6.8284418
time: 3.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6635030, upper bound: 6.6635016
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635005
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5648884, upper bound: 6.5648889
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5648884, upper bound: 6.5648889
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321265, upper bound: 6.8320969
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8321298, upper bound: 6.8320944
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8186359, upper bound: 6.8186141
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8186471, upper bound: 6.8185973
time: 2.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8185975, upper bound: 6.8186456
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8185951, upper bound: 6.8186471
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6481901, upper bound: 6.6482107
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6481901, upper bound: 6.6482107
time: 2.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6696455, upper bound: 6.6696260
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6696414, upper bound: 6.6696326
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6527381, upper bound: 6.6527510
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6527380, upper bound: 6.6527510
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6550229, upper bound: 6.6550262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6550228, upper bound: 6.6550253
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6315562, upper bound: 6.6315501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6315534, upper bound: 6.6315544
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7986861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7986861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8201689, upper bound: 6.8201809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8201733, upper bound: 6.8201759
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8554550, upper bound: 6.8554242
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8554456, upper bound: 6.8554353
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7087561, upper bound: 6.7087601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7087561, upper bound: 6.7087601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489018
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489018
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7539886, upper bound: 6.7539809
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.7539886, upper bound: 6.7539809
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4790513, upper bound: 6.4790513
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4790513, upper bound: 6.4790513
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.3923435, upper bound: 6.3923464
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.3923465, upper bound: 6.3923435
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4081632, upper bound: 6.4081630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4081638, upper bound: 6.4081624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4742590, upper bound: 6.4742643
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4742642, upper bound: 6.4742604
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4918163, upper bound: 6.4918171
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.4918171, upper bound: 6.4918163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5705275, upper bound: 6.5705298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5705275, upper bound: 6.5705298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011903, upper bound: 6.6011965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011906, upper bound: 6.6011916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011916, upper bound: 6.6011902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011884, upper bound: 6.6011914
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5705400, upper bound: 6.5705263
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5705400, upper bound: 6.5705263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011902, upper bound: 6.6011926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6011902, upper bound: 6.6011926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8284469, upper bound: 6.8284423
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8284482, upper bound: 6.8284418
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6635030, upper bound: 6.6635016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6635050, upper bound: 6.6635005
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5648884, upper bound: 6.5648889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.5648884, upper bound: 6.5648889
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8321265, upper bound: 6.8320969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8321298, upper bound: 6.8320944
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8186359, upper bound: 6.8186141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8186471, upper bound: 6.8185973
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8185975, upper bound: 6.8186456
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.8185951, upper bound: 6.8186471
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6481901, upper bound: 6.6482107
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -6.6481901, upper bound: 6.6482107

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5942556, upper bound: 6.5942208
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5942556, upper bound: 6.5942208
time: 2.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5942510, upper bound: 6.5942265
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5942510, upper bound: 6.5942265
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6150127, upper bound: 6.6150305
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6150176, upper bound: 6.6150283
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6229576, upper bound: 6.6229717
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6229576, upper bound: 6.6229722
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550229, upper bound: 6.6550257
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6550222, upper bound: 6.6550262
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6175478, upper bound: 6.6175508
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6175502, upper bound: 6.6175494
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6017772, upper bound: 6.6017731
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6017773, upper bound: 6.6017732
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6315534, upper bound: 6.6315348
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6315395, upper bound: 6.6315544
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7987079, upper bound: 6.7986604
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7986854, upper bound: 6.7986861
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201678, upper bound: 6.8201809
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201689, upper bound: 6.8201797
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201733, upper bound: 6.8201755
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8201726, upper bound: 6.8201759
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8554146, upper bound: 6.8553472
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8553781, upper bound: 6.8553843
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8413459, upper bound: 6.8413379
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8413507, upper bound: 6.8413360
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087561, upper bound: 6.7087575
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087559, upper bound: 6.7087601
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087550, upper bound: 6.7087459
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7087389, upper bound: 6.7087589
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219782
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5219848, upper bound: 6.5219926
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220087, upper bound: 6.5219926
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5220050, upper bound: 6.5219926
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4461685, upper bound: 6.4461781
time: 2.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4461772, upper bound: 6.4461703
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489006, upper bound: 6.4489018
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489012
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7539886, upper bound: 6.7539710
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7539848, upper bound: 6.7539809
time: 2.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7379217, upper bound: 6.7379113
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7379113, upper bound: 6.7379114
time: 3.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489018, upper bound: 6.4489012
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4489013, upper bound: 6.4489017
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4790432, upper bound: 6.4790513
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4790513, upper bound: 6.4790432
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3898131, upper bound: 6.3898200
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3898163, upper bound: 6.3898154
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3549292, upper bound: 6.3549272
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3549289, upper bound: 6.3549280
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192
1: -2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467
2: -3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980
3: -4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634
4: -5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445
5: -4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435
6: -4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641
7: -3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932
8: -5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192
9: -3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1804.42 seconds
