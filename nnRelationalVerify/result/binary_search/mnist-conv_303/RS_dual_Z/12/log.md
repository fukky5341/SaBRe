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
execution time: IAR + LP analysis = 14.13 + 34.18 = 48.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.4199218, upper bound: 2.4199192


# Binary Search by BASE starts (time budget: 3551.68 seconds, max iter: 100)

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
Binary search time: 237.88 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3313.81 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822895, upper bound: 1.8811660
time: 10.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8811661, upper bound: 1.8822893
time: 5.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.02
Output dim: 0, lower bound: -1.8822895, upper bound: 1.8811660
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.02
Output dim: 0, lower bound: -1.8811661, upper bound: 1.8822893

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6700726, 3.6660814
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1047487, 3.1111870
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8081932, 2.8006015
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6870852, 2.6947308
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4973068, 2.4979649
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5617294, 2.5647666
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7623391, 2.7600460
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8774767, 2.8809447

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822743, upper bound: 1.8620990
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8632225, upper bound: 1.8811517
time: 5.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6660824, 3.6700721
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1111870, 3.1047492
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8006010, 2.8081932
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6947317, 2.6870852
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4979649, 2.4973063
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5647669, 2.5617297
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7600455, 2.7623391
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8809443, 2.8774772

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8811511, upper bound: 1.8632223
time: 16.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8620993, upper bound: 1.8822740
time: 13.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 44.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.33
Output dim: 0, lower bound: -1.8822743, upper bound: 1.8620990
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.33
Output dim: 0, lower bound: -1.8632225, upper bound: 1.8811517
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.33
Output dim: 0, lower bound: -1.8811511, upper bound: 1.8632223
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.33
Output dim: 0, lower bound: -1.8620993, upper bound: 1.8822740

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6695681, 3.6594429
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1045361, 3.1084199
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8060064, 2.8004298
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6870661, 2.6944778
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4938641, 2.4976997
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5577202, 2.5644581
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7620449, 2.7562652
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8763361, 2.8808570

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768560, upper bound: 1.8578064
time: 12.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768554, upper bound: 1.8578059
time: 10.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6634331, 3.6655774
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1019821, 3.1109743
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8080215, 2.7984142
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6868315, 2.6947122
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4970417, 2.4945226
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5614209, 2.5607572
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7585583, 2.7597513
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8773890, 2.8798032

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578042, upper bound: 1.8768579
time: 10.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578036, upper bound: 1.8768580
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6655769, 3.6634331
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1109743, 3.1019821
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7984142, 2.8080215
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6947126, 2.6868320
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4945221, 2.4970412
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5607572, 2.5614209
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7597513, 2.7585583
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8798037, 2.8773894

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768581, upper bound: 1.8578033
time: 11.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768581, upper bound: 1.8578039
time: 12.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6594429, 3.6695681
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1084204, 3.1045365
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8004293, 2.8060060
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6944780, 2.6870663
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4976997, 2.4938636
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5644579, 2.5577202
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7562656, 2.7620444
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8808565, 2.8763356

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578063, upper bound: 1.8768552
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578063, upper bound: 1.8768560
time: 6.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8768560, upper bound: 1.8578064
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8768554, upper bound: 1.8578059
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8578042, upper bound: 1.8768579
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8578036, upper bound: 1.8768580
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8768581, upper bound: 1.8578033
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8768581, upper bound: 1.8578039
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8578063, upper bound: 1.8768552
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.55
Output dim: 0, lower bound: -1.8578063, upper bound: 1.8768560

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6715813, 3.6594381
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1045332, 3.1102228
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8048887, 2.7964869
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6870651, 2.6952701
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4938602, 2.4999652
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5579190, 2.5644584
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7620440, 2.7562928
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8770976, 2.8808551

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696871, upper bound: 1.8577997
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768472, upper bound: 1.8506402
time: 12.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6695633, 3.6614552
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1063395, 3.1084166
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8020630, 2.7993126
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6878567, 2.6944761
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4961300, 2.4976954
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5577207, 2.5642600
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7620153, 2.7562647
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8763328, 2.8800898

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696865, upper bound: 1.8578001
time: 10.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768466, upper bound: 1.8506380
time: 8.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6654472, 3.6655731
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1019793, 3.1127772
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8069038, 2.7944713
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6868305, 2.6955042
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4970360, 2.4967880
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5616193, 2.5607574
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7585573, 2.7597795
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8781505, 2.8798013

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506362, upper bound: 1.8768489
time: 12.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577978, upper bound: 1.8696911
time: 11.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6634283, 3.6675901
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1037855, 3.1109710
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8040781, 2.7972975
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6876221, 2.6947103
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4993076, 2.4945178
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5614214, 2.5605590
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7585306, 2.7597513
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8773875, 2.8790359

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506356, upper bound: 1.8768512
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577972, upper bound: 1.8696892
time: 6.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6675901, 3.6634283
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1109715, 3.1037850
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7972975, 2.8040786
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6947098, 2.6876225
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4945183, 2.4993067
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5605593, 2.5614214
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7597513, 2.7585297
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8790355, 2.8773875

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577969
time: 12.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506355
time: 7.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6655731, 3.6654468
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1127777, 3.1019788
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7944708, 2.8069038
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6955032, 2.6868303
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4967880, 2.4970369
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5607576, 2.5616193
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7597799, 2.7585578
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8798003, 2.8781509

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577975
time: 13.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506361
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6614552, 3.6695633
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1084175, 3.1063390
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7993126, 2.8020630
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6944752, 2.6878567
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4976959, 2.4961295
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5642600, 2.5577204
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7562647, 2.7620163
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8800902, 2.8763337

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768466
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696864
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6594381, 3.6715813
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1102228, 3.1045332
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7964869, 2.8048887
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6952705, 2.6870646
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4999657, 2.4938598
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5644584, 2.5579185
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7562933, 2.7620444
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8808551, 2.8770971

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768472
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696870
time: 6.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8696871, upper bound: 1.8577997
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8768472, upper bound: 1.8506402
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8696865, upper bound: 1.8578001
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8768466, upper bound: 1.8506380
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8506362, upper bound: 1.8768489
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8577978, upper bound: 1.8696911
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8506356, upper bound: 1.8768512
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8577972, upper bound: 1.8696892
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577969
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506355
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577975
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506361
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768466
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696864
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768472
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.77
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696870

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6704588, 3.6563339
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1035414, 3.1074414
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8027887, 2.7957425
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6846786, 2.6885743
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4932261, 2.4981761
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5578971, 2.5644000
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7591152, 2.7480450
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8718619, 2.8789849

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696868, upper bound: 1.8576431
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8695306, upper bound: 1.8577993
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6684780, 3.6583157
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1017513, 3.1092315
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8041439, 2.7943878
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6803679, 2.6928840
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4920702, 2.4993315
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5578599, 2.5644369
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7537956, 2.7533622
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8752265, 2.8756189

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768469, upper bound: 1.8504815
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8766906, upper bound: 1.8506375
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6684408, 3.6583514
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1053476, 3.1056352
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7999640, 2.7985687
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6854701, 2.6877801
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4954958, 2.4959064
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5576987, 2.5642014
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7590866, 2.7480168
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8710971, 2.8782201

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8696862, upper bound: 1.8576433
time: 19.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8695299, upper bound: 1.8577993
time: 7.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6664591, 3.6603327
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1035576, 3.1074257
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8013182, 2.7972136
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6811614, 2.6920900
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4943399, 2.4970613
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5576620, 2.5642385
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7537689, 2.7533340
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8744636, 2.8748536

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8768462, upper bound: 1.8504817
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8766900, upper bound: 1.8506378
time: 10.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6643238, 3.6624689
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1009874, 3.1099954
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8048048, 2.7937269
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6844440, 2.6888084
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4964037, 2.4949989
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5615983, 2.5606990
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7556267, 2.7515316
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8729148, 2.8779311

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506359, upper bound: 1.8766927
time: 12.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8504797, upper bound: 1.8768485
time: 17.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6623421, 3.6644502
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0991974, 3.1117854
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8061600, 2.7923717
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6801353, 2.6931183
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4952478, 2.4961543
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5615611, 2.5607362
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7503109, 2.7568493
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8762813, 2.8745651

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577975, upper bound: 1.8695324
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8576412, upper bound: 1.8696887
time: 13.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6623058, 3.6644859
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1027937, 3.1081891
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8019791, 2.7965527
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6852374, 2.6880145
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4986734, 2.4927287
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5613999, 2.5605006
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7555981, 2.7515035
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8721519, 2.8771663

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8506353, upper bound: 1.8766926
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8504790, upper bound: 1.8768484
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.6603241, 3.6664677
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1010027, 3.1099792
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.8033342, 2.7951980
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.6809268, 2.6923242
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4975176, 2.4938841
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.5613627, 2.5605378
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.7502823, 2.7568212
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.8755183, 2.8737998

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577968, upper bound: 1.8695327
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8576406, upper bound: 1.8696886
time: 5.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8696868, upper bound: 1.8576431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8695306, upper bound: 1.8577993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8768469, upper bound: 1.8504815
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8766906, upper bound: 1.8506375
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8696862, upper bound: 1.8576433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8695299, upper bound: 1.8577993
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8768462, upper bound: 1.8504817
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8766900, upper bound: 1.8506378
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8506359, upper bound: 1.8766927
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8504797, upper bound: 1.8768485
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8577975, upper bound: 1.8695324
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8576412, upper bound: 1.8696887
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8506353, upper bound: 1.8766926
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8504790, upper bound: 1.8768484
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8577968, upper bound: 1.8695327
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.66
Output dim: 0, lower bound: -1.8576406, upper bound: 1.8696886
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577969
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506355
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8696892, upper bound: 1.8577975
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8768493, upper bound: 1.8506361
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768466
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696864
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8506383, upper bound: 1.8768472
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.66
Output dim: 0, lower bound: -1.8577999, upper bound: 1.8696870
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
type: RSZ, layer: 1, pos: 5844
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5844

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7597223, upper bound: 1.7582452
time: 14.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7582454, upper bound: 1.7597223
time: 5.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.73
Output dim: 0, lower bound: -1.7597223, upper bound: 1.7582452
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.73
Output dim: 0, lower bound: -1.7582454, upper bound: 1.7597223

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5111351, 3.5077152
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0172129, 3.0227313
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7069969, 2.7004900
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5886908, 2.5952437
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4171448, 2.4177089
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4762917, 2.4788949
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8469744, 3.8392811
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6797366, 2.6777711
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7699556, 2.7729278

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7597097, upper bound: 1.7419058
time: 12.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7433808, upper bound: 1.7582327
time: 7.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5077152, 3.5111351
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0227318, 3.0172133
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7004900, 2.7069969
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5952444, 2.5886903
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4177094, 2.4171443
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4788952, 2.4762919
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8392811, 3.8469744
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6777711, 2.6797366
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7729273, 2.7699552

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 500

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7582329, upper bound: 1.7433802
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7419040, upper bound: 1.7597089
time: 14.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 37.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.34
Output dim: 0, lower bound: -1.7597097, upper bound: 1.7419058
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 37.34
Output dim: 0, lower bound: -1.7433808, upper bound: 1.7582327
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.34
Output dim: 0, lower bound: -1.7582329, upper bound: 1.7433802
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 37.34
Output dim: 0, lower bound: -1.7419040, upper bound: 1.7597089

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5097551, 3.5010762
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0166359, 3.0199647
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7048101, 2.7000303
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5886374, 2.5949907
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4137020, 2.4169898
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4722824, 2.4780576
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8466959, 3.8379402
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6789436, 2.6739902
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7688131, 2.7726893

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7540143, upper bound: 1.7376867
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7540138, upper bound: 1.7376867
time: 6.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5044966, 3.5063343
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0144463, 3.0221539
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7065372, 2.6983027
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5884371, 2.5951915
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4164257, 2.4142661
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4754543, 2.4748855
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8456335, 3.8390026
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6759558, 2.6769786
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7697172, 2.7717862

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7376854, upper bound: 1.7540155
time: 11.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7376849, upper bound: 1.7540153
time: 19.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5063343, 3.5044966
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0221539, 3.0144463
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6983023, 2.7065372
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5951910, 2.5884371
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4142666, 2.4164252
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4748855, 2.4754543
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8390026, 3.8456335
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6769791, 2.6759558
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7717867, 2.7697167

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7540159, upper bound: 1.7376846
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7540159, upper bound: 1.7376851
time: 6.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5010757, 3.5097547
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0199642, 3.0166359
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7000303, 2.7048097
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5949907, 2.5886378
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169903, 2.4137020
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4780579, 2.4722824
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8379412, 3.8466959
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6739902, 2.6789441
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7726889, 2.7688136

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7376870, upper bound: 1.7540134
time: 11.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7376870, upper bound: 1.7540139
time: 11.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 38.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7540143, upper bound: 1.7376867
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7540138, upper bound: 1.7376867
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7376854, upper bound: 1.7540155
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7376849, upper bound: 1.7540153
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7540159, upper bound: 1.7376846
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7540159, upper bound: 1.7376851
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7376870, upper bound: 1.7540134
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.35
Output dim: 0, lower bound: -1.7376870, upper bound: 1.7540139

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5114803, 3.5010715
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0166321, 3.0215092
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7032881, 2.6960874
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5886364, 2.5956693
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4136982, 2.4189310
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4724526, 2.4780579
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8433142, 3.8291950
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6789446, 2.6740141
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7694659, 2.7726874

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7477450, upper bound: 1.7376815
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7540068, upper bound: 1.7314522
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5097504, 3.5028005
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0181808, 3.0199614
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7008667, 2.6985097
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5893154, 2.5949888
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4156437, 2.4169855
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4722829, 2.4778879
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8379498, 3.8345537
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6789198, 2.6739898
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7688117, 2.7720313

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7477448, upper bound: 1.7376814
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7540063, upper bound: 1.7314517
time: 9.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5062218, 3.5063300
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0144434, 3.0236983
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7050161, 2.6943598
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5884342, 2.5958703
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4164219, 2.4162078
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4756246, 2.4748857
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8422518, 3.8302565
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6759558, 2.6770024
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7703700, 2.7717838

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7314506, upper bound: 1.7540101
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7376799, upper bound: 1.7477460
time: 16.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5044918, 3.5080590
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0159912, 3.0221505
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7025938, 2.6967821
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5891151, 2.5951896
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4183674, 2.4142623
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4754548, 2.4747157
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8368883, 3.8356152
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6759310, 2.6769781
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7697139, 2.7711282

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7314504, upper bound: 1.7540082
time: 8.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7376794, upper bound: 1.7477480
time: 12.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5080585, 3.5044918
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0221510, 3.0159912
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6967821, 2.7025948
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5951900, 2.5891144
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4142628, 2.4183669
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4747157, 2.4754548
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8356152, 3.8368883
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6769781, 2.6759315
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7711272, 2.7697148

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7477462, upper bound: 1.7376794
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7540084, upper bound: 1.7314502
time: 11.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5063305, 3.5062218
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0236988, 3.0144429
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6943598, 2.7050166
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5958710, 2.5884352
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4162083, 2.4164209
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4748859, 2.4756246
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8302574, 3.8422518
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6770029, 2.6759553
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7717834, 2.7703695

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7477462, upper bound: 1.7376798
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7540084, upper bound: 1.7314504
time: 11.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5028000, 3.5097504
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0199614, 3.0181804
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6985092, 2.7008667
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5949879, 2.5893154
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4169846, 2.4156432
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4778876, 2.4722826
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8345537, 3.8379498
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6739893, 2.6789193
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7720313, 2.7688117

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7314518, upper bound: 1.7540062
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7376815, upper bound: 1.7477446
time: 8.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5010719, 3.5114803
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0215092, 3.0166326
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.6960869, 2.7032890
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5956688, 2.5886362
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4189320, 2.4136977
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4780579, 2.4724524
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8291950, 3.8433142
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6740141, 2.6789441
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7726874, 2.7694659

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7314518, upper bound: 1.7540067
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7376815, upper bound: 1.7477447
time: 8.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 32.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7477450, upper bound: 1.7376815
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7540068, upper bound: 1.7314522
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7477448, upper bound: 1.7376814
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7540063, upper bound: 1.7314517
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7314506, upper bound: 1.7540101
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7376799, upper bound: 1.7477460
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7314504, upper bound: 1.7540082
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7376794, upper bound: 1.7477480
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7477462, upper bound: 1.7376794
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7540084, upper bound: 1.7314502
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7477462, upper bound: 1.7376798
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7540084, upper bound: 1.7314504
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7314518, upper bound: 1.7540062
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7376815, upper bound: 1.7477446
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7314518, upper bound: 1.7540067
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.74
Output dim: 0, lower bound: -1.7376815, upper bound: 1.7477447

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262
1: -21.6256638, -17.3819923, -21.6256638, -17.3819923, -3.5048161, 3.5032263
2: -5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.0131960, 3.0209169
3: -14.0028372, -10.9323034, -14.0028372, -10.9323034, -2.7029171, 2.6934214
4: -9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.5854335, 2.5891743
5: -7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.4156218, 2.4144187
6: -5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.4755979, 2.4748273
7: -11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8403454, 3.8259964
8: -4.1027942, -0.9745383, -4.1027942, -0.9745383, -2.6722660, 2.6687546
9: -4.8675470, -1.8201666, -4.8675470, -1.8201666, -2.7651343, 2.7694335

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5821
type: RSZ, layer: 1, pos: 4644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7314503, upper bound: 1.7538223
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7312616, upper bound: 1.7540077
time: 11.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 36.67 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 36.67
Output dim: 0, lower bound: -1.7314503, upper bound: 1.7538223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 36.67
Output dim: 0, lower bound: -1.7312616, upper bound: 1.7540077
Binary search (step 3): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1500.43 seconds
