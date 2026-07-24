## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 0.64869276486
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786)
1: (-0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580)
2: (-0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435)
3: (-0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037)
4: (-0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566)
5: (-0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337)
6: (-0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695)
7: (-0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825)
8: (-0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762)
9: (-0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096)

## BASE Result
execution time: IAR + LP analysis = 1.17 + 2.42 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7714816, upper bound: 0.7714816


# Binary Search by BASE starts (time budget: 2696.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary Search Result
Binary search time: 11.93 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2684.48 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.29
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.29
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.29
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.29
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715403, upper bound: 0.6715425
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.97
Output dim: 0, lower bound: -0.6715421, upper bound: 0.6715407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
time: 0.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486920, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.92
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486918

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
time: 0.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486930
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486927, upper bound: 0.6486916
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486914

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
time: 1.09 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486910, upper bound: 0.6486912
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486908
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486913
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.15
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486897

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
time: 1.50 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486870, upper bound: 0.6486933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486892, upper bound: 0.6486904
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486899
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486900, upper bound: 0.6486896
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.62
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486875

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
time: 1.00 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 10.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456107
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456050, upper bound: 0.6456110
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456105
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456052, upper bound: 0.6456108
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456107, upper bound: 0.6456054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 10.88
Output dim: 0, lower bound: -0.6456103, upper bound: 0.6456054
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715431, upper bound: 0.6715462
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.84
Output dim: 0, lower bound: -0.6715459, upper bound: 0.6715431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
time: 0.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486966
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486954
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486950
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.82
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486941

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
time: 0.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486937, upper bound: 0.6486965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486952
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486946
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486946, upper bound: 0.6486950
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486952, upper bound: 0.6486947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486961, upper bound: 0.6486943
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.10
Output dim: 0, lower bound: -0.6486964, upper bound: 0.6486938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
time: 1.04 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486966
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486920
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486880, upper bound: 0.6486934
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486903
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486879, upper bound: 0.6486959
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486912
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486928
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486898
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486878, upper bound: 0.6486954
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486907
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486882, upper bound: 0.6486917
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486889
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486886, upper bound: 0.6486947
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486898
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486916
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486890
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486966
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486920
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486880, upper bound: 0.6486934
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486903
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486879, upper bound: 0.6486961
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486912
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486928
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486898
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486878, upper bound: 0.6486954
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486907
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486882, upper bound: 0.6486917
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486889
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486886, upper bound: 0.6486947
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486898
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486916
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486890
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486950
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486893
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486924
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486890
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486944
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486886
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486920
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486882
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486941
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486890
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486911
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486883
time: 0.98 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486966
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486920
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486880, upper bound: 0.6486934
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486903
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486879, upper bound: 0.6486959
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486912
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486878, upper bound: 0.6486954
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486907
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486882, upper bound: 0.6486917
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486889
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486886, upper bound: 0.6486947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486916
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486874, upper bound: 0.6486966
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486920
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486880, upper bound: 0.6486934
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486879, upper bound: 0.6486961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486928
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486878, upper bound: 0.6486954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486907
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486882, upper bound: 0.6486917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486889
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486886, upper bound: 0.6486947
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486885, upper bound: 0.6486950
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486893
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486889, upper bound: 0.6486944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486886
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486920
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486893, upper bound: 0.6486941
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486909, upper bound: 0.6486911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486962
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486934, upper bound: 0.6486932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486939, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486917, upper bound: 0.6486954
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486943, upper bound: 0.6486916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486919, upper bound: 0.6486947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486947, upper bound: 0.6486916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486913, upper bound: 0.6486951
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486942, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486914, upper bound: 0.6486946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486949, upper bound: 0.6486921
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486926, upper bound: 0.6486940
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486958, upper bound: 0.6486912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486930, upper bound: 0.6486938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 0, lower bound: -0.6486962, upper bound: 0.6486912
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 0, lower bound: -0.7450595, upper bound: 0.7450595

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
time: 1.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.35
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.35
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.35
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.35
Output dim: 0, lower bound: -0.7386935, upper bound: 0.7386935

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715412, upper bound: 0.6715438
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.98
Output dim: 0, lower bound: -0.6715434, upper bound: 0.6715416

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
time: 0.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486943
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486935
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486929
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486924

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
time: 1.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486941
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486941
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486925
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486933
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486926
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486921

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
time: 1.03 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486918, upper bound: 0.6486920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486922, upper bound: 0.6486917
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486925, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486912, upper bound: 0.6486924
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486916, upper bound: 0.6486922
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486941
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486904
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486934
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486898
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486925
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486894
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486905
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486883
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486941
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486904
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486934
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486898
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486925
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486894
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486905
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486883
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486933
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486883
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486929
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486882
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486894, upper bound: 0.6486908
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486880
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486899, upper bound: 0.6486898
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486880
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486897
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486876
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486933
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486883
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486929
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486882
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486894, upper bound: 0.6486908
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486880
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486899, upper bound: 0.6486898
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486880
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486897
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486876
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.2562878, 1.0549664, 0.2562878, 1.0549664, -0.7986786, 0.7986786
1: -0.2454222, 0.2509358, -0.2454222, 0.2509358, -0.4963580, 0.4963580
2: -0.1680138, 0.3355297, -0.1680138, 0.3355297, -0.5035435, 0.5035435
3: -0.1995975, 0.2523062, -0.1995975, 0.2523062, -0.4519037, 0.4519037
4: -0.2665573, 0.2256994, -0.2665573, 0.2256994, -0.4922566, 0.4922566
5: -0.3110178, 0.3794159, -0.3110178, 0.3794159, -0.6904337, 0.6904337
6: -0.1894653, 0.3079041, -0.1894653, 0.3079041, -0.4973695, 0.4973695
7: -0.2645668, 0.3842157, -0.2645668, 0.3842157, -0.6487825, 0.6487825
8: -0.2571728, 0.3466035, -0.2571728, 0.3466035, -0.6037762, 0.6037762
9: -0.2574216, 0.3313880, -0.2574216, 0.3313880, -0.5888096, 0.5888096

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
time: 1.00 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486904
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486934
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486925
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486905
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486883
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486941
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486904
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486875, upper bound: 0.6486934
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486898
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486925
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486894
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486905
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486933
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486929
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486882
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486894, upper bound: 0.6486908
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486899, upper bound: 0.6486898
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486897
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486881, upper bound: 0.6486933
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486883
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486883, upper bound: 0.6486929
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486894, upper bound: 0.6486908
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486899, upper bound: 0.6486898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486897
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486871, upper bound: 0.6486945
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486910
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486897, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486898, upper bound: 0.6486942
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486906, upper bound: 0.6486935
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486907, upper bound: 0.6486929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486929, upper bound: 0.6486907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486902, upper bound: 0.6486931
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486903, upper bound: 0.6486928
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486931, upper bound: 0.6486910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486938, upper bound: 0.6486900
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.14
Output dim: 0, lower bound: -0.6486941, upper bound: 0.6486901
Binary search (step 2): status=Status.UNKNOWN, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=0.7986786365509033
rel_dist={0: [-0.7714815551950811, 0.7714815551950811]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1670.12 seconds
