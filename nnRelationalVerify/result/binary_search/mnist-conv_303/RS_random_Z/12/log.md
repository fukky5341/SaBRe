## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.7540085754
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262)
1: (-21.6256638, -17.3819923, -21.6256638, -17.3819923, -4.2436714, 4.2436714)
2: (-5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1439934, 3.1439934)
3: (-14.0028372, -10.9323034, -14.0028372, -10.9323034, -3.0705338, 3.0705338)
4: (-9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.9589658, 2.9589658)
5: (-7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.8115454, 2.8115454)
6: (-5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.7522268, 2.7522268)
7: (-11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838)
8: (-4.1027942, -0.9745383, -4.1027942, -0.9745383, -3.1282558, 3.1282558)
9: (-4.8675470, -1.8201666, -4.8675470, -1.8201666, -3.0473804, 3.0473804)

## BASE Result
execution time: IAR + LP analysis = 14.04 + 34.53 = 48.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.4199218, upper bound: 2.4199192


# Binary Search by BASE starts (time budget: 3551.44 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNREACHABLE, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.8390512466430664

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.669461727142334
rel_dist={0: [-1.0059846949496585, 1.0059850175342175]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.754256248474121
rel_dist={0: [-1.1909365767727103, 1.1909392632351246]}

## Binary Search Result
Binary search time: 239.22 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3312.21 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8772283, upper bound: 1.8826283
time: 10.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8826287, upper bound: 1.8772280
time: 8.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.10
Output dim: 0, lower bound: -1.8772283, upper bound: 1.8826283
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.10
Output dim: 0, lower bound: -1.8826287, upper bound: 1.8772280

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6510973, 3.6622949
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1103811, 3.0858030
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8285770, 2.8240271
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6946526, 2.6924558
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4927182, 2.5002618
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5593939, 2.5815406
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7678351, 2.7679968
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8799238, 2.8855438

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8763895, upper bound: 1.8817818
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8763908, upper bound: 1.8817818
time: 13.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6575756, 3.6510978
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0858030, 3.0999594
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8240271, 2.8266459
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6924562, 2.6937246
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4970603, 2.4927182
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5721893, 2.5593939
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7679267, 2.7678351
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8831682, 2.8799243

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8769052, upper bound: 1.8753819
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8807823, upper bound: 1.8715051
time: 6.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 0, lower bound: -1.8763895, upper bound: 1.8817818
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 0, lower bound: -1.8763908, upper bound: 1.8817818
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 0, lower bound: -1.8769052, upper bound: 1.8753819
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 0, lower bound: -1.8807823, upper bound: 1.8715051

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6511011, 3.6622953
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1109457, 3.0857863
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8287497, 2.8240218
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6947746, 2.6924515
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4927135, 2.5004215
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5593772, 2.5821338
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7678437, 2.7679963
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8799219, 2.8856378

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8738853, upper bound: 1.8817780
time: 11.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8763874, upper bound: 1.8792797
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6510983, 3.6622949
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1103640, 3.0858030
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8285713, 2.8240271
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6946487, 2.6924558
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4927182, 2.5002570
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5593939, 2.5815239
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7678342, 2.7679968
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8799238, 2.8855410

Time for backsubstitution: 13.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8763756, upper bound: 1.8627128
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8573239, upper bound: 1.8817646
time: 9.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6577425, 3.6500864
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0859261, 3.0992680
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8240252, 2.8265500
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6926484, 2.6926355
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4971457, 2.4922485
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5721316, 2.5594156
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7681589, 2.7665920
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8825045, 2.8800340

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8744120, upper bound: 1.8753798
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8769032, upper bound: 1.8728860
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6565647, 3.6510978
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0851116, 3.0999594
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8239307, 2.8266459
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6913667, 2.6937246
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4965906, 2.4927182
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5721893, 2.5593362
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7666845, 2.7678351
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8831682, 2.8792615

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8803814, upper bound: 1.8699838
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8792586, upper bound: 1.8711038
time: 6.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8738853, upper bound: 1.8817780
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8763874, upper bound: 1.8792797
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8763756, upper bound: 1.8627128
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8573239, upper bound: 1.8817646
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8744120, upper bound: 1.8753798
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8769032, upper bound: 1.8728860
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8803814, upper bound: 1.8699838
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.74
Output dim: 0, lower bound: -1.8792586, upper bound: 1.8711038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6526132, 3.6644459
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1108494, 3.0856519
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8230219, 2.8199677
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6896954, 2.6852877
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4935236, 2.5009975
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5572047, 2.5790691
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7636938, 2.7650552
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8811378, 2.8864999

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8738850, upper bound: 1.8816212
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8737289, upper bound: 1.8817774
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6532521, 3.6638074
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1108103, 3.0856910
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8246965, 2.8182940
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6876106, 2.6873724
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4932890, 2.5012307
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5563126, 2.5799615
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7649031, 2.7638459
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8807831, 2.8868546

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8706640, upper bound: 1.8774360
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8745411, upper bound: 1.8735585
time: 11.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6505933, 3.6556554
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1101513, 3.0830355
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8263845, 2.8238554
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6946297, 2.6922011
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4892745, 2.4999914
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5553851, 2.5812159
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7675395, 2.7642155
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8787832, 2.8854532

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8759756, upper bound: 1.8611892
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8748531, upper bound: 1.8623126
time: 5.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6444583, 3.6617904
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1075974, 3.0855899
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8283997, 2.8218398
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6943951, 2.6924353
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4924521, 2.4968143
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5590858, 2.5775151
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7640529, 2.7677016
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8798361, 2.8843994

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8570513, upper bound: 1.8817546
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8573138, upper bound: 1.8814920
time: 10.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6592531, 3.6522369
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0858307, 3.0991344
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8182974, 2.8224950
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6875668, 2.6854718
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4979558, 2.4928250
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5699587, 2.5563509
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7640085, 2.7636514
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8837204, 2.8808956

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8735729, upper bound: 1.8753719
time: 15.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8744054, upper bound: 1.8745448
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6598921, 3.6515980
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0857925, 3.0991740
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8199711, 2.8208213
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6854839, 2.6875563
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4977231, 2.4930577
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5690665, 2.5572433
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7652178, 2.7624426
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8833656, 2.8812509

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8760545, upper bound: 1.8720446
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8760545, upper bound: 1.8720433
time: 7.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6690602, 3.6596036
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0899010, 3.1111870
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8054781, 2.8006015
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6847267, 2.6947308
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4968376, 2.4936237
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5617294, 2.5519140
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7610970, 2.7599545
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8774767, 2.8770390

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8733614, upper bound: 1.8668595
time: 22.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8803705, upper bound: 1.8668441
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6650701, 3.6635942
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0963392, 3.1047492
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7978868, 2.8081932
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6923733, 2.6870852
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4974957, 2.4929647
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5647669, 2.5488768
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7588034, 2.7622476
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8809443, 2.8735714

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8792583, upper bound: 1.8709476
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8790899, upper bound: 1.8711038
time: 9.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8738850, upper bound: 1.8816212
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8737289, upper bound: 1.8817774
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8706640, upper bound: 1.8774360
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8745411, upper bound: 1.8735585
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8759756, upper bound: 1.8611892
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8748531, upper bound: 1.8623126
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8570513, upper bound: 1.8817546
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8573138, upper bound: 1.8814920
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8735729, upper bound: 1.8753719
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8744054, upper bound: 1.8745448
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8760545, upper bound: 1.8720446
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8760545, upper bound: 1.8720433
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8733614, upper bound: 1.8668595
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8803705, upper bound: 1.8668441
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8792583, upper bound: 1.8709476
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.47
Output dim: 0, lower bound: -1.8790899, upper bound: 1.8711038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6507044, 3.6617532
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1090193, 3.0843539
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8259687, 2.8222184
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6976757, 2.6957841
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4975052, 2.5062308
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5657964, 2.5856018
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7666440, 2.7691455
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8822098, 2.8864942

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8738847, upper bound: 1.8816210
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8738847, upper bound: 1.8791270
time: 10.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6499205, 3.6625381
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1095514, 3.0838213
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8252726, 2.8229146
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.7001915, 2.6932688
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4987564, 2.5049801
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5637374, 2.5876606
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7677827, 2.7680054
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8811321, 2.8875713

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8733571, upper bound: 1.8802541
time: 19.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8722362, upper bound: 1.8813772
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6534176, 3.6627960
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1109352, 3.0850015
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8246937, 2.8181973
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6878033, 2.6862831
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4933763, 2.5007610
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5562544, 2.5799828
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7651348, 2.7626042
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8801193, 2.8869643

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8702635, upper bound: 1.8759375
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8691414, upper bound: 1.8770573
time: 11.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6522398, 3.6638074
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1101208, 3.0856910
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8245993, 2.8182940
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6865215, 2.6873724
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4928193, 2.5012307
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5563126, 2.5799031
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7636604, 2.7638459
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8807831, 2.8861918

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8742685, upper bound: 1.8735487
time: 11.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8745310, upper bound: 1.8732857
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6630907, 3.6641622
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1149416, 3.0942636
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8079319, 2.7978115
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6879897, 2.6932077
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4895229, 2.5008969
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5449257, 2.5737932
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7619524, 2.7563353
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8730917, 2.8832312

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8705573, upper bound: 1.8568965
time: 16.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8705567, upper bound: 1.8568965
time: 7.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6591005, 3.6681523
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1213799, 3.0878258
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8003397, 2.8054032
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6956353, 2.6855621
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4901810, 2.5002384
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5479627, 2.5707560
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7596598, 2.7586284
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8765593, 2.8797631

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8705594, upper bound: 1.8568937
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8705594, upper bound: 1.8568942
time: 9.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6434131, 3.6640439
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1056681, 3.0897455
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8329611, 2.8197217
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6994061, 2.6901386
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4964752, 2.4949360
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5584269, 2.5789351
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7636933, 2.7684779
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8821168, 2.8833361

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8498834, upper bound: 1.8817456
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8570450, upper bound: 1.8745877
time: 10.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6444583, 3.6607451
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1075974, 3.0836601
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8262825, 2.8218398
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6920981, 2.6924353
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4905748, 2.4968143
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5590858, 2.5768554
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7640529, 2.7673416
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8787732, 2.8843994

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8522393, upper bound: 1.8764177
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8522393, upper bound: 1.8764177
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6582289, 3.6531639
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0854549, 3.0994749
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8150473, 2.8254423
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6867132, 2.6862407
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4980116, 2.4927635
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5710220, 2.5551751
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7630100, 2.7645540
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8869958, 2.8772750

Time for backsubstitution: 14.48 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.068826198577881
rel_dist={0: [-1.882689891073806, 1.882689645160882]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.9238462448120117
rel_dist={0: [-1.4958789978991227, 1.4958786753824898]}

## Binary search (step 2) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=5, k_high=6, k_mid=5, eps_mid=0.0195312, abs_max=3.008641242980957
rel_dist={0: [-1.6317345307279227, 1.631736453304411]}

## Binary search (step 3) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7590378, upper bound: 1.7599587
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7599568, upper bound: 1.7590377
time: 6.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.73
Output dim: 0, lower bound: -1.7590378, upper bound: 1.7599587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.73
Output dim: 0, lower bound: -1.7599568, upper bound: 1.7590377

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4981842, 3.4998584
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0120487, 3.0126629
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7232838, 2.7285957
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5944762, 2.5958672
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169369, 2.4168367
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4874954, 2.4855762
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8711042, 3.8684454
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6846528, 2.6862822
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7779365, 2.7720256

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7535573, upper bound: 1.7598975
time: 11.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589780, upper bound: 1.7544762
time: 12.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4992085, 3.4981842
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0124226, 3.0120482
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7265339, 2.7232838
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5953307, 2.5944760
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4168367, 2.4168978
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4855762, 2.4867516
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8684454, 3.8700762
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6856523, 2.6846538
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7720256, 2.7756457

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7544761, upper bound: 1.7589780
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7598977, upper bound: 1.7535569
time: 5.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 0, lower bound: -1.7535573, upper bound: 1.7598975
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 0, lower bound: -1.7589780, upper bound: 1.7544762
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 0, lower bound: -1.7544761, upper bound: 1.7589780
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 0, lower bound: -1.7598977, upper bound: 1.7535569

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4917059, 3.5029774
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0189581, 2.9985061
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7245646, 2.7259769
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5950899, 2.5945981
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4125948, 2.4189606
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4747000, 2.4917636
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8582506
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6845622, 2.6863294
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7746930, 2.7735996

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7514457, upper bound: 1.7598979
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7535556, upper bound: 1.7577857
time: 10.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4981842, 3.4933796
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9978914, 3.0126629
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7206650, 2.7285957
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5932074, 2.5958672
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169369, 2.4124947
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4874954, 2.4727807
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8609095, 3.8684454
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6846528, 2.6861906
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7779365, 2.7687826

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535481
time: 11.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535467
time: 11.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4927311, 3.5013032
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0193338, 2.9978919
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7278156, 2.7206650
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5959444, 2.5932069
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4124947, 2.4190221
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4727802, 2.4929390
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8734303, 3.8598814
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6855607, 2.6847005
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7687821, 2.7772193

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7481581, upper bound: 1.7589703
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7544685, upper bound: 1.7526854
time: 7.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4992085, 3.4917059
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9982672, 3.0120482
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7239151, 2.7232838
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5940609, 2.5944760
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4168367, 2.4125562
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4855762, 2.4739559
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8582506, 3.8700762
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6856523, 2.6845622
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7720256, 2.7724023

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535474
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535459
time: 7.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7514457, upper bound: 1.7598979
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7535556, upper bound: 1.7577857
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535481
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535467
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7481581, upper bound: 1.7589703
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7544685, upper bound: 1.7526854
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535474
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.17
Output dim: 0, lower bound: -1.7589557, upper bound: 1.7535459

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4932179, 3.5050368
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0188570, 2.9983716
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7188373, 2.7216840
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5897117, 2.5874331
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4133711, 2.4195375
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4723997, 2.4886985
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8582630
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6804132, 2.6832166
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7758584, 2.7744603

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7514345, upper bound: 1.7589558
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7514360, upper bound: 1.7589537
time: 7.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4937654, 3.5044894
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0188236, 2.9984050
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7202725, 2.7202497
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5879254, 2.5892200
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4131718, 2.4197373
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4716344, 2.4894633
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8588619
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6814489, 2.6821799
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7755551, 2.7747650

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7533137, upper bound: 1.7560676
time: 9.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7518366, upper bound: 1.7575441
time: 24.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4981871, 3.4933801
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9983778, 3.0126462
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7208138, 2.7285905
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5933104, 2.5958638
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169312, 2.4126325
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4874778, 2.4732912
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8613367, 3.8684320
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6846604, 2.6861901
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7779326, 2.7688632

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589554, upper bound: 1.7533630
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7587707, upper bound: 1.7535477
time: 8.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4981842, 3.4933796
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9978743, 3.0126629
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7206593, 2.7285957
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5932026, 2.5958672
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169369, 2.4124904
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4874954, 2.4727640
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8608961, 3.8684454
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6846528, 2.6861906
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7779365, 2.7687798

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7571686, upper bound: 1.7535462
time: 13.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589556, upper bound: 1.7517582
time: 11.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4913259, 3.4981999
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0180855, 2.9951100
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7257166, 2.7197270
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5929432, 2.5865116
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4116969, 2.4172330
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4727550, 2.4928811
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8715229, 3.8556194
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6818690, 2.6764522
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7635455, 2.7748675

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 500

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7460341, upper bound: 1.7589690
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7481564, upper bound: 1.7568588
time: 9.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4896274, 3.4998980
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0165520, 2.9966445
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7268782, 2.7185659
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5892487, 2.5902057
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4107060, 2.4182229
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4727230, 2.4929128
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8691692, 3.8579741
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6773124, 2.6810093
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7664313, 2.7719827

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7526814, upper bound: 1.7526832
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7544684, upper bound: 1.7509234
time: 9.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4992113, 3.4917059
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9987555, 3.0120320
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7240639, 2.7232785
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5941639, 2.5944726
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4168320, 2.4126940
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4855585, 2.4744663
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8586760, 3.8700628
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6856580, 2.6845617
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7720218, 2.7724829

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7526613, upper bound: 1.7535400
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589481, upper bound: 1.7472335
time: 8.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4992094, 3.4917059
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9982519, 3.0120482
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7239094, 2.7232838
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5940571, 2.5944760
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4168367, 2.4125514
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4855762, 2.4739394
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8582373, 3.8700762
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6856523, 2.6845622
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7720256, 2.7723994

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7540336, upper bound: 1.7527254
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7581352, upper bound: 1.7486234
time: 6.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7514345, upper bound: 1.7589558
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7514360, upper bound: 1.7589537
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7533137, upper bound: 1.7560676
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7518366, upper bound: 1.7575441
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7589554, upper bound: 1.7533630
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7587707, upper bound: 1.7535477
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7571686, upper bound: 1.7535462
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7589556, upper bound: 1.7517582
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7460341, upper bound: 1.7589690
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7481564, upper bound: 1.7568588
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7526814, upper bound: 1.7526832
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7544684, upper bound: 1.7509234
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7526613, upper bound: 1.7535400
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7589481, upper bound: 1.7472335
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7540336, upper bound: 1.7527254
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.12
Output dim: 0, lower bound: -1.7581352, upper bound: 1.7486234

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4932199, 3.5050373
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0193367, 2.9983544
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7189856, 2.7216792
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5898161, 2.5874300
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4133663, 2.4196734
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4723830, 2.4892044
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8582497
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6804199, 2.6832161
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7758555, 2.7745404

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7514218, upper bound: 1.7426119
time: 12.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7350929, upper bound: 1.7589412
time: 21.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4932179, 3.5050368
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0188398, 2.9983716
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7188330, 2.7216840
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5897093, 2.5874331
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4133711, 2.4195323
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4723997, 2.4886818
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8582630
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6804123, 2.6832166
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7758584, 2.7744570

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7463583, upper bound: 1.7538774
time: 16.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7463583, upper bound: 1.7538774
time: 15.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5056930, 3.5129967
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0236120, 3.0087118
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7007351, 2.6942048
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5812855, 2.5891337
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4134183, 2.4205484
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4611754, 2.4816072
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8530006, 3.8280687
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6755347, 2.6742997
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7698641, 2.7720470

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7530603, upper bound: 1.7560589
time: 12.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7533049, upper bound: 1.7558155
time: 15.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5022731, 3.5164170
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0291300, 3.0031939
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6942272, 2.7007117
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5878382, 2.5825801
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4139829, 2.4199839
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4637785, 2.4790039
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8453074, 3.8357620
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6735692, 2.6762652
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7728357, 2.7690744

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7469142, upper bound: 1.7567238
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7510161, upper bound: 1.7526219
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4961662, 3.4906874
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9965477, 3.0112715
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7236624, 2.7308407
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6012917, 2.6060011
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4209156, 2.4176879
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4957738, 2.4798217
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8629990, 3.8705435
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6876111, 2.6901169
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7788525, 2.7688589

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7568438, upper bound: 1.7533613
time: 19.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7589537, upper bound: 1.7512510
time: 21.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4954939, 3.4913602
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9970045, 3.0108151
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7230654, 2.7314377
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6034489, 2.6038451
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4219875, 2.4166155
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4940090, 2.4815865
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8634481, 3.8700943
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6885877, 2.6891394
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7779293, 2.7697821

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7538486, upper bound: 1.7527270
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7579502, upper bound: 1.7486274
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.4979897, 3.4915438
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -2.9951200, 3.0089889
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7072153, 2.7185159
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5701632, 2.5651488
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4134350, 2.4108601
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4857039, 2.4703779
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8633823, 3.8716393
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6677456, 2.6735106
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7673264, 2.7609682

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 907

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7569271, upper bound: 1.7518273
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7554462, upper bound: 1.7533042
time: 7.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7514218, upper bound: 1.7426119
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7350929, upper bound: 1.7589412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7463583, upper bound: 1.7538774
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7463583, upper bound: 1.7538774
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7530603, upper bound: 1.7560589
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7533049, upper bound: 1.7558155
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7469142, upper bound: 1.7567238
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7510161, upper bound: 1.7526219
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7568438, upper bound: 1.7533613
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7589537, upper bound: 1.7512510
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7538486, upper bound: 1.7527270
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7579502, upper bound: 1.7486274
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7569271, upper bound: 1.7518273
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.61
Output dim: 0, lower bound: -1.7554462, upper bound: 1.7533042
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7589556, upper bound: 1.7517582
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7460341, upper bound: 1.7589690
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7481564, upper bound: 1.7568588
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7544684, upper bound: 1.7509234
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7589481, upper bound: 1.7472335
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7540336, upper bound: 1.7527254
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.61
Output dim: 0, lower bound: -1.7581352, upper bound: 1.7486234
Binary search (step 3): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 1726.83 seconds
