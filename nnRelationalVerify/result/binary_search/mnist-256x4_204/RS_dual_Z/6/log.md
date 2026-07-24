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
execution time: IAR + LP analysis = 1.47 + 4.08 = 5.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -7.1077352, upper bound: 7.1077352


# Binary Search by BASE starts (time budget: 2694.46 seconds, max iter: 100)

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
Binary search time: 24.39 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2670.07 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 2.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.22
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.22
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278335
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278335
time: 1.91 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278335, upper bound: 6.5278448
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278335, upper bound: 6.5278448
time: 1.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278335
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278335
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 6, lower bound: -6.5278335, upper bound: 6.5278448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 6, lower bound: -6.5278335, upper bound: 6.5278448

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
time: 1.40 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
time: 1.42 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
time: 1.27 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
time: 1.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450327
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450312

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
time: 1.46 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
time: 1.44 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
time: 1.45 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
time: 1.44 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
time: 1.37 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
time: 1.37 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
time: 1.37 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
time: 1.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450312, upper bound: 6.3450303
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450327
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450327, upper bound: 6.3450303
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 6, lower bound: -6.3450303, upper bound: 6.3450312

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
time: 1.62 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
time: 1.52 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
time: 1.81 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
time: 1.55 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
time: 1.85 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
time: 1.53 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
time: 1.60 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
time: 1.32 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
time: 1.35 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
time: 1.60 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
time: 1.32 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
time: 1.61 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293445
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293499, upper bound: 6.3293444
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293504
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293449, upper bound: 6.3293447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293447, upper bound: 6.3293449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293504, upper bound: 6.3293444
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293444, upper bound: 6.3293499
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.03
Output dim: 6, lower bound: -6.3293445, upper bound: 6.3293444

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
time: 1.65 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
time: 1.51 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
time: 1.46 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
time: 1.65 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
time: 1.66 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
time: 1.51 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
time: 1.45 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
time: 1.68 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
time: 1.65 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
time: 1.51 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
time: 1.45 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
time: 1.64 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
time: 1.65 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
time: 1.44 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
time: 1.64 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
time: 1.78 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
time: 1.57 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
time: 1.45 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
time: 1.52 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
time: 1.78 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
time: 1.55 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
time: 1.43 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
time: 1.44 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
time: 1.79 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
time: 1.54 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
time: 1.45 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
time: 1.44 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
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

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
time: 1.55 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
time: 1.59 seconds

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

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
time: 1.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 10.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842520
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842498
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842521, upper bound: 6.2842416
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842431, upper bound: 6.2842517
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842481, upper bound: 6.2842480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842525, upper bound: 6.2842452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842528, upper bound: 6.2842414
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842414, upper bound: 6.2842528
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842452, upper bound: 6.2842525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842480, upper bound: 6.2842481
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842517, upper bound: 6.2842431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842416, upper bound: 6.2842521
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842473, upper bound: 6.2842492
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842498, upper bound: 6.2842449
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.84
Output dim: 6, lower bound: -6.2842520, upper bound: 6.2842415

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842415, upper bound: 6.2842327
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842227, upper bound: 6.2842520
time: 1.40 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842449, upper bound: 6.2842323
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842205, upper bound: 6.2842498
time: 1.37 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842492, upper bound: 6.2842244
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2842315, upper bound: 6.2842473
time: 1.71 seconds

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

Time for backsubstitution: 1.18 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
time: 3.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.07
Output dim: 6, lower bound: -7.1071524, upper bound: 7.1071524
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.07
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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270923
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270923
time: 1.49 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5270923, upper bound: 6.5270972
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5270923, upper bound: 6.5270972
time: 1.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.64
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270923
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.64
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270923
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.64
Output dim: 6, lower bound: -6.5270923, upper bound: 6.5270972
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.64
Output dim: 6, lower bound: -6.5270923, upper bound: 6.5270972

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
time: 1.35 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
time: 1.34 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
time: 1.31 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
time: 1.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443574
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443563

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
time: 1.43 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
time: 1.53 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
time: 1.43 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
time: 1.53 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
time: 1.38 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
time: 1.34 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
time: 1.38 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
time: 1.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443563, upper bound: 6.3443525
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443527, upper bound: 6.3443574
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443574, upper bound: 6.3443527
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.00
Output dim: 6, lower bound: -6.3443525, upper bound: 6.3443563

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
time: 1.36 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
time: 1.46 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
time: 1.36 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
time: 1.46 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
time: 1.36 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
time: 1.46 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
time: 1.54 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
time: 1.46 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
time: 1.39 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
time: 1.43 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
time: 1.45 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
time: 1.42 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
time: 1.39 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
time: 1.43 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
time: 1.45 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
time: 1.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286699, upper bound: 6.3286688
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286714, upper bound: 6.3286684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286723
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286705
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286705, upper bound: 6.3286688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286723, upper bound: 6.3286684
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286684, upper bound: 6.3286714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.16
Output dim: 6, lower bound: -6.3286688, upper bound: 6.3286699

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
time: 1.40 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
time: 1.40 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
time: 1.62 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
time: 1.49 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
time: 1.42 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
time: 1.39 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
time: 1.46 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
time: 1.49 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
time: 1.40 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
time: 1.52 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
time: 1.46 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
time: 1.48 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
time: 1.42 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
time: 1.52 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
time: 1.46 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
time: 1.48 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
time: 1.51 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
time: 1.56 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
time: 1.50 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
time: 1.59 seconds

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

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
time: 1.73 seconds

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

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
time: 1.72 seconds

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

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
time: 1.50 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
time: 1.39 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
time: 1.59 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
time: 1.55 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
time: 1.43 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
time: 1.42 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
time: 1.54 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
time: 1.71 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
time: 1.44 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
time: 1.41 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 10.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835916, upper bound: 6.2835958
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835935, upper bound: 6.2835941
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835947, upper bound: 6.2835944
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835959, upper bound: 6.2835917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835929, upper bound: 6.2835957
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835954, upper bound: 6.2835939
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835961, upper bound: 6.2835938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835915, upper bound: 6.2835969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835938, upper bound: 6.2835961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835939, upper bound: 6.2835954
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835957, upper bound: 6.2835929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835917, upper bound: 6.2835959
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835944, upper bound: 6.2835947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835941, upper bound: 6.2835935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.31
Output dim: 6, lower bound: -6.2835958, upper bound: 6.2835916

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835821
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835856, upper bound: 6.2835915
time: 1.31 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2835969, upper bound: 6.2835821
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2835856, upper bound: 6.2835915
time: 1.31 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1069232, upper bound: 7.1069232
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1069232, upper bound: 7.1069232
time: 3.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 6, lower bound: -7.1069232, upper bound: 7.1069232
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 6, lower bound: -7.1069232, upper bound: 7.1069232

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5256910, upper bound: 6.5256891
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5256910, upper bound: 6.5256891
time: 1.51 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5256891, upper bound: 6.5256910
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5256891, upper bound: 6.5256910
time: 1.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 6, lower bound: -6.5256910, upper bound: 6.5256891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 6, lower bound: -6.5256910, upper bound: 6.5256891
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 6, lower bound: -6.5256891, upper bound: 6.5256910
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 6, lower bound: -6.5256891, upper bound: 6.5256910

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
time: 1.67 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
time: 1.66 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
time: 1.45 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
time: 1.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435836
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.17
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435831

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
time: 3.51 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
time: 2.48 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
time: 3.51 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
time: 2.49 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
time: 2.03 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
time: 2.04 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
time: 2.07 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
time: 2.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435831, upper bound: 6.3435815
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435836
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435836, upper bound: 6.3435815
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.99
Output dim: 6, lower bound: -6.3435815, upper bound: 6.3435831

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
time: 2.07 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
time: 1.67 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
time: 2.03 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
time: 1.66 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
time: 2.05 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
time: 1.65 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
time: 2.04 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
time: 1.68 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
time: 1.94 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
time: 1.76 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
time: 2.06 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
time: 1.73 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
time: 1.96 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
time: 1.70 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
time: 1.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279204, upper bound: 6.3279202
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279210, upper bound: 6.3279198
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279211
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279203, upper bound: 6.3279206
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279206, upper bound: 6.3279203
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279211, upper bound: 6.3279198
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279198, upper bound: 6.3279210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.96
Output dim: 6, lower bound: -6.3279202, upper bound: 6.3279204

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829278
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.50 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829276, upper bound: 6.2829269
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829291, upper bound: 6.2829259
time: 1.86 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829285
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.71 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829275, upper bound: 6.2829273
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829283, upper bound: 6.2829257
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829278
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.55 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829276, upper bound: 6.2829269
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829291, upper bound: 6.2829259
time: 1.85 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829285
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.72 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829275, upper bound: 6.2829273
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829283, upper bound: 6.2829257
time: 1.64 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829278
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.50 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829276, upper bound: 6.2829269
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829291, upper bound: 6.2829259
time: 1.86 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829285
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.70 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829275, upper bound: 6.2829273
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829283, upper bound: 6.2829257
time: 1.64 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829278
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.49 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829276, upper bound: 6.2829269
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829291, upper bound: 6.2829259
time: 1.84 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829260, upper bound: 6.2829285
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829274
time: 1.75 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829275, upper bound: 6.2829273
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829283, upper bound: 6.2829257
time: 1.64 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829257, upper bound: 6.2829283
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829273, upper bound: 6.2829275
time: 1.73 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829285, upper bound: 6.2829260
time: 1.59 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829259, upper bound: 6.2829291
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829269, upper bound: 6.2829276
time: 1.45 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829260
time: 1.43 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829257, upper bound: 6.2829283
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829273, upper bound: 6.2829275
time: 1.51 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829285, upper bound: 6.2829260
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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829259, upper bound: 6.2829291
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829269, upper bound: 6.2829276
time: 1.40 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829260
time: 1.55 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829257, upper bound: 6.2829283
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829273, upper bound: 6.2829275
time: 1.60 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829285, upper bound: 6.2829260
time: 1.59 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829259, upper bound: 6.2829291
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829269, upper bound: 6.2829276
time: 1.43 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829278, upper bound: 6.2829260
time: 1.44 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829257, upper bound: 6.2829283
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829273, upper bound: 6.2829275
time: 1.54 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829274, upper bound: 6.2829278
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829285, upper bound: 6.2829260
time: 1.54 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829259, upper bound: 6.2829291
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2829269, upper bound: 6.2829276
time: 1.39 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1803.54 seconds
