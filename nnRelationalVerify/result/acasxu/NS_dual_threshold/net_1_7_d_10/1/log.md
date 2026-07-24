## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.722369459961


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-47.8403511, 184.0399475, -47.8403511, 184.0399475, -231.8802948, 231.8802948)
1: (-126.9594421, 423.5780640, -126.9594421, 423.5780640, -550.5374756, 550.5374756)
2: (-182.2965851, 372.3862000, -182.2965851, 372.3862000, -554.6828003, 554.6828003)
3: (-108.4022675, 446.6286011, -108.4022675, 446.6286011, -555.0307617, 555.0307617)
4: (-168.5674896, 322.6039429, -168.5674896, 322.6039429, -491.1714172, 491.1714172)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.00 + 1.92 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7280013

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7279964
time: 0.72 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7280013
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.58 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7279964
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7280013

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -46.3404083, 178.2005768, -45.5786934, 175.2346954, -221.5751038, 223.7792664
1: -123.0632172, 410.1153870, -121.0812378, 403.2799683, -526.3431396, 531.1966553
2: -177.1840057, 360.1800537, -174.5699463, 354.0225830, -531.2066040, 534.7499390
3: -105.1081772, 432.6358032, -103.4265671, 425.5083008, -530.6163940, 536.0623169
4: -163.6906281, 312.0748291, -161.1979370, 306.7643738, -470.4550171, 473.2727661

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7276716, upper bound: 187.7276934
time: 0.67 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7276312, upper bound: 187.7276243
time: 0.77 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -46.6620522, 179.4360657, -46.8357430, 180.1120148, -226.7740479, 226.2718048
1: -124.0822144, 412.7102661, -124.5590820, 414.7407837, -538.8229980, 537.2693481
2: -178.8888855, 362.3389893, -179.6586914, 363.4034424, -542.2923584, 541.9976807
3: -105.9738235, 435.5882263, -106.3493500, 437.5282288, -543.5020752, 541.9375610
4: -165.1683960, 313.9503174, -165.8073578, 314.8886414, -480.0570374, 479.7576904

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279843
time: 0.65 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279272
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.55 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -187.7276716, upper bound: 187.7276934
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -187.7276312, upper bound: 187.7276243
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279843
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279272

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -44.8479996, 172.5733032, -41.5381317, 159.4344940, -204.2824860, 214.1114349
1: -118.9563599, 397.5036011, -110.2674408, 365.7463379, -484.7026978, 507.7710571
2: -170.6914520, 349.1826172, -159.1838837, 322.6174316, -493.3088989, 508.3664856
3: -101.5797653, 419.1305237, -94.1774445, 386.3644104, -487.9441833, 513.3078613
4: -157.8497925, 302.5675354, -146.9637146, 279.5232239, -437.3730164, 449.5311890

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_B1

### Relational analysis result of NS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275414, upper bound: 187.7275390
time: 0.80 seconds

## Relational analysis of NS_B1_B1_B2

### Relational analysis result of NS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275669, upper bound: 187.7276251
time: 0.74 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -45.2153511, 173.9443054, -43.7549019, 168.3105774, -213.5258942, 217.6992035
1: -120.0141068, 400.3981018, -116.1500244, 387.3979187, -507.4120178, 516.5480957
2: -172.5856018, 351.8429565, -167.2028046, 340.4636536, -513.0492554, 519.0457153
3: -102.4687500, 422.1174622, -99.1577377, 408.3130493, -510.7817993, 521.2752075
4: -159.4995728, 304.7919617, -154.4586945, 294.9054260, -454.4049988, 459.2506714

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275484, upper bound: 187.7275113
time: 1.15 seconds

## Relational analysis of NS_B1_B2_B2

### Relational analysis result of NS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275511, upper bound: 187.7275292
time: 0.82 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -43.6637306, 167.8074493, -45.2125320, 173.8206787, -217.4844055, 213.0199890
1: -116.1008453, 386.0664062, -120.2461929, 400.3527222, -516.4534912, 506.3125610
2: -168.0746460, 338.6493530, -173.8504791, 350.5428772, -518.6175537, 512.4998169
3: -99.2385864, 407.6341553, -102.7224045, 422.4079590, -521.6465454, 510.3565369
4: -155.0011902, 293.5181885, -160.3375702, 303.8022156, -458.8034058, 453.8557739

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267370, upper bound: 187.7270009
time: 0.77 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7278827, upper bound: 187.7279201
time: 0.67 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -46.6296539, 178.9131775, -45.2281532, 173.6696167, -220.2992706, 224.1413269
1: -124.0993576, 411.2045593, -120.3959808, 399.5948792, -523.6942139, 531.6005249
2: -180.1733093, 361.0405884, -174.5735321, 350.0928345, -530.2661133, 535.6141357
3: -106.0504761, 434.2114868, -102.8537903, 421.8562317, -527.9066772, 537.0652466
4: -165.9819336, 312.7274170, -160.8480988, 303.3878174, -469.3697205, 473.5755005

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7279273, upper bound: 187.7279272
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279273
time: 0.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.56 seconds
NS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7275414, upper bound: 187.7275390
NS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7275669, upper bound: 187.7276251
NS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7275484, upper bound: 187.7275113
NS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7275511, upper bound: 187.7275292
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7267370, upper bound: 187.7270009
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7278827, upper bound: 187.7279201
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7279273, upper bound: 187.7279272
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -187.7279272, upper bound: 187.7279273

## BFS NS instance: NS_B1_B1_B1

### Backsubstitution after applying NS history:
0: -43.2409554, 166.4004974, -39.5790596, 151.9988861, -195.2398224, 205.9795532
1: -114.6586227, 383.4612732, -104.9415283, 349.0539856, -463.7126160, 488.4028015
2: -164.6087036, 336.8103638, -151.2744751, 308.0357666, -472.6444092, 488.0848389
3: -97.9295273, 404.3211060, -89.6290894, 368.4991455, -466.4286804, 493.9501953
4: -152.2020721, 291.8877563, -139.7333679, 266.9154053, -419.1174927, 431.6210632

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
time: 0.66 seconds

## Relational analysis of NS_B1_B1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
time: 1.04 seconds

## BFS NS instance: NS_B1_B1_B2

### Backsubstitution after applying NS history:
0: -43.9155922, 168.9880829, -40.3670120, 155.0805054, -198.9960938, 209.3550720
1: -116.5222549, 389.1702576, -107.2317276, 356.4628296, -472.9850769, 496.4019165
2: -167.3626404, 341.8352661, -154.6642151, 313.1611938, -480.5237732, 496.4994202
3: -99.5170441, 410.4484253, -91.5844498, 376.5230408, -476.0400391, 502.0328674
4: -154.7113953, 296.2523193, -142.7593994, 271.3388062, -426.0501404, 439.0116577

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
time: 0.74 seconds

## Relational analysis of NS_B1_B1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7276251
time: 0.69 seconds

## BFS NS instance: NS_B1_B2_B1

### Backsubstitution after applying NS history:
0: -43.6727524, 168.0110168, -41.4450569, 159.4190521, -203.0917664, 209.4560699
1: -115.9052734, 386.8268127, -109.9310150, 367.1177673, -483.0230103, 496.7577209
2: -166.8669586, 339.9036865, -158.4000397, 322.8183594, -489.6853027, 498.3037109
3: -98.9833984, 407.8333435, -93.8630524, 386.8898010, -485.8731995, 501.6964111
4: -154.1594696, 294.5001526, -146.2857971, 279.6833496, -433.8427734, 440.7859497

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274664, upper bound: 187.7275000
time: 0.68 seconds

## Relational analysis of NS_B1_B2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274664, upper bound: 187.7275113
time: 1.03 seconds

## BFS NS instance: NS_B1_B2_B2

### Backsubstitution after applying NS history:
0: -44.2643623, 170.2778320, -43.2720642, 166.5788879, -210.8432312, 213.5498810
1: -117.5149918, 391.8777466, -114.9974594, 382.9272156, -500.4421997, 506.8751831
2: -169.1766052, 344.3340454, -165.8288574, 336.3431091, -505.5197144, 510.1628113
3: -100.3553009, 413.2550049, -98.1179657, 403.8304749, -504.1857605, 511.3729553
4: -156.2838593, 298.3448181, -153.0210419, 291.3052368, -447.5890808, 451.3658447

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
time: 0.75 seconds

## Relational analysis of NS_B1_B2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274779, upper bound: 187.7275292
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -42.0196381, 161.4741364, -42.8678398, 164.7687531, -206.7883759, 204.3419495
1: -111.7107468, 371.6521606, -114.0224075, 379.7088318, -491.4194946, 485.6745605
2: -161.9454041, 325.8217468, -165.3869476, 332.1094971, -494.0548401, 491.2086792
3: -95.5211258, 392.4644470, -97.4912415, 400.8524475, -496.3735352, 489.9556580
4: -149.2867432, 282.4783936, -152.3906555, 287.9379578, -437.2247009, 434.8690491

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266682, upper bound: 187.7270009
time: 0.60 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266682, upper bound: 187.7270009
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -42.8359566, 164.6016388, -45.1999588, 173.8377991, -216.6737366, 209.8015900
1: -113.9013290, 378.6423035, -120.5114594, 400.3252258, -514.2265625, 499.1536865
2: -165.0507812, 332.1534119, -174.2127533, 349.8500366, -514.9006958, 506.3661499
3: -97.3774261, 399.8705444, -102.8613052, 422.9147644, -520.2921753, 502.7318420
4: -152.1631317, 287.9348755, -160.5251007, 303.1787109, -455.3418579, 448.4599609

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274894, upper bound: 187.7276917
time: 1.03 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274894, upper bound: 187.7279201
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.6296539, 178.9131775, -44.1546440, 169.7290497, -216.3587036, 223.0677948
1: -124.0993576, 411.2045593, -117.4306717, 391.0097046, -515.1090698, 528.6352539
2: -180.1733093, 361.0405884, -170.0241394, 342.1984558, -522.3717651, 531.0646973
3: -106.0504761, 434.2114868, -100.3515778, 412.6105957, -518.6610107, 534.5630493
4: -165.9819336, 312.7274170, -156.7424927, 296.6056519, -462.5875549, 469.4699097

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275861, upper bound: 187.7272884
time: 1.39 seconds

## Relational analysis of NS_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267175, upper bound: 187.7269302
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7278827, upper bound: 187.7278827
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -46.6296539, 178.9131775, -50.0622749, 192.0670013, -238.6966553, 228.9754333
1: -124.0993576, 411.2045593, -133.4757080, 441.5740662, -565.6734009, 544.6802979
2: -180.1733093, 361.0405884, -193.7078552, 386.6830444, -566.8563232, 554.7484131
3: -106.0504761, 434.2114868, -114.0948105, 466.8600464, -572.9105225, 548.3062744
4: -165.9819336, 312.7274170, -178.3865356, 335.0679626, -501.0498657, 491.1139526

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275556, upper bound: 187.7274155
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273398, upper bound: 187.7273398
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.22 seconds
NS_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
NS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
NS_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
NS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7276251
NS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274664, upper bound: 187.7275000
NS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274664, upper bound: 187.7275113
NS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
NS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274779, upper bound: 187.7275292
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7266682, upper bound: 187.7270009
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7266682, upper bound: 187.7270009
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274894, upper bound: 187.7276917
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7274894, upper bound: 187.7279201
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7267175, upper bound: 187.7269302
NS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7278827, upper bound: 187.7278827
NS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7275556, upper bound: 187.7274155
NS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -187.7273398, upper bound: 187.7273398

## BFS NS instance: NS_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -42.2604561, 162.6482086, -39.5790596, 151.9988861, -194.2593231, 202.2272644
1: -112.0051422, 374.9786072, -104.9415283, 349.0539856, -461.0591431, 479.9201355
2: -160.7832489, 329.3633728, -151.2744751, 308.0357666, -468.8190002, 480.6378479
3: -95.6769104, 395.3126831, -89.6290894, 368.4991455, -464.1760559, 484.9417725
4: -148.6735992, 285.4589539, -139.7333679, 266.9154053, -415.5889893, 425.1923218

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
time: 0.78 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -43.8708611, 168.8724518, -39.5790596, 151.9988861, -195.8697510, 208.4515076
1: -116.4157639, 388.6484375, -104.9415283, 349.0539856, -465.4697571, 493.5899658
2: -166.9455109, 341.2932739, -151.2744751, 308.0357666, -474.9812622, 492.5677490
3: -99.3476639, 409.9704590, -89.6290894, 368.4991455, -467.8468018, 499.5995483
4: -154.3255463, 295.6994629, -139.7333679, 266.9154053, -421.2409668, 435.4328308

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
time: 0.74 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -42.2604561, 162.6482086, -40.3670120, 155.0805054, -197.3409576, 203.0152130
1: -112.0051422, 374.9786072, -107.2317276, 356.4628296, -468.4679565, 482.2102661
2: -160.7832489, 329.3633728, -154.6642151, 313.1611938, -473.9444580, 484.0275574
3: -95.6769104, 395.3126831, -91.5844498, 376.5230408, -472.1999512, 486.8971252
4: -148.6735992, 285.4589539, -142.7593994, 271.3388062, -420.0123901, 428.2183228

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_B2_A1_A1

### Relational analysis result of NS_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275769
time: 1.16 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275769
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -43.8708611, 168.8724518, -40.3670120, 155.0805054, -198.9513702, 209.2394562
1: -116.4157639, 388.6484375, -107.2317276, 356.4628296, -472.8785706, 495.8801270
2: -166.9455109, 341.2932739, -154.6642151, 313.1611938, -480.1066895, 495.9574280
3: -99.3476639, 409.9704590, -91.5844498, 376.5230408, -475.8706360, 501.5549011
4: -154.3255463, 295.6994629, -142.7593994, 271.3388062, -425.6643677, 438.4588318

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_B2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275629
time: 0.82 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275631
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -42.7306976, 164.3975372, -41.4450569, 159.4190521, -202.1497498, 205.8425903
1: -113.3704453, 378.5969849, -109.9310150, 367.1177673, -480.4882202, 488.5278931
2: -163.2726440, 332.7073975, -158.4000397, 322.8183594, -486.0910034, 491.1074219
3: -96.8319702, 399.1424255, -93.8630524, 386.8898010, -483.7217712, 493.0054932
4: -150.8275909, 288.3114929, -146.2857971, 279.6833496, -430.5109253, 434.5972900

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_B1_A1_A1

### Relational analysis result of NS_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275000
time: 0.73 seconds

## Relational analysis of NS_B1_B2_B1_A1_A2

### Relational analysis result of NS_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275000
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -44.5705299, 171.5597229, -41.4450569, 159.4190521, -203.9895782, 213.0047760
1: -118.4292221, 394.4607239, -109.9310150, 367.1177673, -485.5469971, 504.3916626
2: -170.5802307, 346.4087219, -158.4000397, 322.8183594, -493.3985901, 504.8087769
3: -101.0553284, 416.0765076, -93.8630524, 386.8898010, -487.9450989, 509.9395752
4: -157.4567413, 300.0660400, -146.2857971, 279.6833496, -437.1400757, 446.3518372

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_B1_A2_A1

### Relational analysis result of NS_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275113
time: 1.05 seconds

## Relational analysis of NS_B1_B2_B1_A2_A2

### Relational analysis result of NS_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275113
time: 0.73 seconds

## BFS NS instance: NS_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -42.7306976, 164.3975372, -43.2720642, 166.5788879, -209.3095856, 207.6696014
1: -113.3704453, 378.5969849, -114.9974594, 382.9272156, -496.2976685, 493.5943604
2: -163.2726440, 332.7073975, -165.8288574, 336.3431091, -499.6157532, 498.5362244
3: -96.8319702, 399.1424255, -98.1179657, 403.8304749, -500.6624451, 497.2603760
4: -150.8275909, 288.3114929, -153.0210419, 291.3052368, -442.1328125, 441.3325195

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_B2_A1_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275210
time: 0.71 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2

### Relational analysis result of NS_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275210
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -44.5705299, 171.5597229, -43.2720642, 166.5788879, -211.1494141, 214.8317566
1: -118.4292221, 394.4607239, -114.9974594, 382.9272156, -501.3564453, 509.4581299
2: -170.5802307, 346.4087219, -165.8288574, 336.3431091, -506.9233398, 512.2375488
3: -101.0553284, 416.0765076, -98.1179657, 403.8304749, -504.8857422, 514.1944580
4: -157.4567413, 300.0660400, -153.0210419, 291.3052368, -448.7619324, 453.0870972

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_B2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275273
time: 0.79 seconds

## Relational analysis of NS_B1_B2_B2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275273
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.0661240, 157.8053284, -42.8678398, 164.7687531, -205.8348694, 200.6731567
1: -109.1240540, 363.3471069, -114.0224075, 379.7088318, -488.8328552, 477.3695068
2: -158.2162018, 318.5402527, -165.3869476, 332.1094971, -490.3256836, 483.9271851
3: -93.3242035, 383.6660461, -97.4912415, 400.8524475, -494.1766052, 481.1572571
4: -145.8542480, 276.2144470, -152.3906555, 287.9379578, -433.7922058, 428.6051025

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7259992
time: 0.91 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7261335
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -42.8238945, 164.5812073, -42.8678398, 164.7687531, -207.5926514, 207.4490356
1: -114.0151596, 378.5130615, -114.0224075, 379.7088318, -493.7239990, 492.5354614
2: -165.2616425, 331.5993958, -165.3869476, 332.1094971, -497.3711243, 496.9862671
3: -97.3868561, 399.8609314, -97.4912415, 400.8524475, -498.2392578, 497.3521423
4: -152.1722412, 287.3803711, -152.3906555, 287.9379578, -440.1101379, 439.7709351

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260339
time: 0.74 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7261578
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.0661240, 157.8053284, -45.1999588, 173.8377991, -214.9039154, 203.0052795
1: -109.1240540, 363.3471069, -120.5114594, 400.3252258, -509.4492798, 483.8585205
2: -158.2162018, 318.5402527, -174.2127533, 349.8500366, -508.0661926, 492.7529907
3: -93.3242035, 383.6660461, -102.8613052, 422.9147644, -516.2389526, 486.5273438
4: -145.8542480, 276.2144470, -160.5251007, 303.1787109, -449.0329590, 436.7395630

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260389
time: 0.82 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7271933
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -42.8238945, 164.5812073, -45.1999588, 173.8377991, -216.6616974, 209.7811584
1: -114.0151596, 378.5130615, -120.5114594, 400.3252258, -514.3403931, 499.0244751
2: -165.2616425, 331.5993958, -174.2127533, 349.8500366, -515.1116333, 505.8121338
3: -97.3868561, 399.8609314, -102.8613052, 422.9147644, -520.3015747, 502.7222290
4: -152.1722412, 287.3803711, -160.5251007, 303.1787109, -455.3508911, 447.9054565

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260741
time: 0.98 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7277043
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -45.2675362, 173.6474915, -41.9552193, 161.2417297, -206.5092621, 215.6027069
1: -120.4328613, 399.2705383, -111.5608521, 371.6982422, -492.1311035, 510.8313904
2: -175.1054840, 350.4068604, -161.9490662, 325.0089722, -500.1144409, 512.3559570
3: -102.9632950, 421.6907043, -95.4081497, 392.4071960, -495.3704529, 517.0988770
4: -161.2548523, 303.5817261, -149.1902313, 281.8039856, -443.0588379, 452.7719421

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7266682
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7274894
time: 0.98 seconds

## BFS NS instance: NS_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -45.6311836, 175.0916290, -44.2351913, 170.1514893, -215.7826691, 219.3268127
1: -121.4478989, 402.3750610, -117.9425812, 391.9692383, -513.4171143, 520.3175659
2: -176.3440094, 353.3907166, -170.6576538, 342.3324890, -518.6764526, 524.0483398
3: -103.7844238, 424.8986816, -100.6752853, 414.0962524, -517.8806763, 525.5739746
4: -162.4336243, 306.1250610, -157.1894226, 296.6888428, -459.1224670, 463.3144531

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7267370
time: 1.29 seconds

## Relational analysis of NS_B2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7278827
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -44.8710709, 172.2863922, -45.2416077, 173.3715210, -218.2425690, 217.5279999
1: -119.2514038, 396.4571838, -120.6988602, 398.2498474, -517.5012207, 517.1560059
2: -172.0989227, 348.2472534, -174.9852600, 348.9457092, -521.0446167, 523.2325439
3: -101.8511810, 418.3502197, -103.1604919, 421.7271729, -523.5783691, 521.5107422
4: -158.9090118, 301.6211243, -161.1642761, 302.4093933, -461.3184204, 462.7853699

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_B1_B1

### Relational analysis result of NS_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275497, upper bound: 187.7274109
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B2_B1_B2

### Relational analysis result of NS_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273280, upper bound: 187.7273255
time: 0.64 seconds

## BFS NS instance: NS_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -45.6103783, 175.0097809, -48.3729210, 185.5788422, -231.1892242, 223.3826904
1: -121.3288345, 402.2579041, -128.9464874, 426.5923462, -547.9212036, 531.2044067
2: -176.1827087, 353.3314819, -187.3093567, 373.8816223, -550.0643311, 540.6408691
3: -103.6820908, 424.5975037, -110.1913681, 450.7362976, -554.4182739, 534.7888794
4: -162.3066864, 306.0259094, -172.4528503, 323.9251404, -486.2318115, 478.4787598

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266764, upper bound: 187.7267210
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273049, upper bound: 187.7273049
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.65 seconds
NS_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
NS_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
NS_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
NS_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275390
NS_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275769
NS_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275769
NS_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275629
NS_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275631
NS_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275000
NS_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275000
NS_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275113
NS_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275113
NS_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275210
NS_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275210
NS_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275273
NS_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7274774, upper bound: 187.7275273
NS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7259992
NS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7261335
NS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260339
NS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7261578
NS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260389
NS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7271933
NS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7253978, upper bound: 187.7260741
NS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7254180, upper bound: 187.7277043
NS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7266682
NS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7274894
NS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7267370
NS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7270009, upper bound: 187.7278827
NS_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7275497, upper bound: 187.7274109
NS_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7273280, upper bound: 187.7273255
NS_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7266764, upper bound: 187.7267210
NS_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -187.7273049, upper bound: 187.7273049

## BFS NS instance: NS_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -41.6508636, 160.2781372, -39.5790596, 151.9988861, -193.6497498, 199.8571930
1: -110.3976593, 369.5987854, -104.9415283, 349.0539856, -459.4516296, 474.5402832
2: -158.6191254, 324.4791260, -151.2744751, 308.0357666, -466.6548767, 475.7535706
3: -94.3115692, 389.6538696, -89.6290894, 368.4991455, -462.8107300, 479.2829590
4: -146.6292725, 281.2406616, -139.7333679, 266.9154053, -413.5446777, 420.9740295

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
time: 0.79 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275116
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -42.5479698, 163.8011322, -39.5790596, 151.9988861, -194.5468597, 203.3801880
1: -112.9720078, 377.9263306, -104.9415283, 349.0539856, -462.0259705, 482.8678589
2: -162.3658295, 330.9160461, -151.2744751, 308.0357666, -470.4014893, 482.1905212
3: -96.4609375, 398.5251770, -89.6290894, 368.4991455, -464.9600830, 488.1542358
4: -150.0088654, 286.8193054, -139.7333679, 266.9154053, -416.9242554, 426.5526733

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
time: 0.61 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275116
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -43.1604233, 166.1354828, -39.5790596, 151.9988861, -195.1592865, 205.7145386
1: -114.5572968, 382.3229370, -104.9415283, 349.0539856, -463.6112671, 487.2644653
2: -164.4700317, 335.5992126, -151.2744751, 308.0357666, -472.5057678, 486.8736572
3: -97.7738724, 403.3337402, -89.6290894, 368.4991455, -466.2730103, 492.9627991
4: -151.9817657, 290.7724304, -139.7333679, 266.9154053, -418.8971558, 430.5057983

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B1_A2_A1_B1

### Relational analysis result of NS_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275390
time: 0.78 seconds

## Relational analysis of NS_B1_B1_B1_A2_A1_B2

### Relational analysis result of NS_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275164, upper bound: 187.7275293
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -44.6245079, 171.5823822, -39.5790596, 151.9988861, -196.6233826, 211.1614380
1: -118.6856537, 395.1393433, -104.9415283, 349.0539856, -467.7396240, 500.0808716
2: -170.3513336, 346.5853882, -151.2744751, 308.0357666, -478.3870239, 497.8598633
3: -101.2509537, 416.9933777, -89.6290894, 368.4991455, -469.7500916, 506.6224670
4: -157.3656311, 300.2826538, -139.7333679, 266.9154053, -424.2810059, 440.0160217

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B1_A2_A2_B1

### Relational analysis result of NS_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275390
time: 0.67 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_B2

### Relational analysis result of NS_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275164, upper bound: 187.7275293
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -41.6508636, 160.2781372, -40.3670120, 155.0805054, -196.7313690, 200.6451416
1: -110.3976593, 369.5987854, -107.2317276, 356.4628296, -466.8604431, 476.8304138
2: -158.6191254, 324.4791260, -154.6642151, 313.1611938, -471.7803345, 479.1432495
3: -94.3115692, 389.6538696, -91.5844498, 376.5230408, -470.8345947, 481.2383118
4: -146.6292725, 281.2406616, -142.7593994, 271.3388062, -417.9680786, 424.0000610

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B2_A1_A1_B1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
time: 0.68 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274895, upper bound: 187.7275407
time: 0.94 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -42.5479698, 163.8011322, -40.3670120, 155.0805054, -197.6284790, 204.1681519
1: -112.9720078, 377.9263306, -107.2317276, 356.4628296, -469.4348145, 485.1580200
2: -162.3658295, 330.9160461, -154.6642151, 313.1611938, -475.5270081, 485.5802307
3: -96.4609375, 398.5251770, -91.5844498, 376.5230408, -472.9839783, 490.1096191
4: -150.0088654, 286.8193054, -142.7593994, 271.3388062, -421.3476562, 429.5786743

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
time: 0.84 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274895, upper bound: 187.7275407
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -43.1604233, 166.1354828, -40.3670120, 155.0805054, -198.2409058, 206.5025024
1: -114.5572968, 382.3229370, -107.2317276, 356.4628296, -471.0201111, 489.5545959
2: -164.4700317, 335.5992126, -154.6642151, 313.1611938, -477.6312256, 490.2633972
3: -97.7738724, 403.3337402, -91.5844498, 376.5230408, -474.2969055, 494.9181824
4: -151.9817657, 290.7724304, -142.7593994, 271.3388062, -423.3205566, 433.5318298

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B2_A2_A1_B1

### Relational analysis result of NS_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275310, upper bound: 187.7275629
time: 0.71 seconds

## Relational analysis of NS_B1_B1_B2_A2_A1_B2

### Relational analysis result of NS_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275311, upper bound: 187.7275444
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -44.6245079, 171.5823822, -40.3670120, 155.0805054, -199.7050171, 211.9493866
1: -118.6856537, 395.1393433, -107.2317276, 356.4628296, -475.1484985, 502.3710022
2: -170.3513336, 346.5853882, -154.6642151, 313.1611938, -483.5125122, 501.2495117
3: -101.2509537, 416.9933777, -91.5844498, 376.5230408, -477.7739868, 508.5778198
4: -157.3656311, 300.2826538, -142.7593994, 271.3388062, -428.7044373, 443.0420532

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275310, upper bound: 187.7275631
time: 0.88 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275444
time: 0.91 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -42.1634254, 162.1604309, -41.4450569, 159.4190521, -201.5824738, 203.6054688
1: -111.8679581, 373.4281616, -109.9310150, 367.1177673, -478.9857178, 483.3591003
2: -161.2144623, 328.1668701, -158.4000397, 322.8183594, -484.0328369, 486.5668945
3: -95.5358963, 393.7258301, -93.8630524, 386.8898010, -482.4256897, 487.5888672
4: -148.8794556, 284.3475037, -146.2857971, 279.6833496, -428.5628052, 430.6333008

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B1_A1_A1_B1

### Relational analysis result of NS_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259318, upper bound: 187.7260912
time: 0.99 seconds

## Relational analysis of NS_B1_B2_B1_A1_A1_B2

### Relational analysis result of NS_B1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266851, upper bound: 187.7266318
time: 0.79 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -43.5457916, 167.4363098, -41.4450569, 159.4190521, -202.9648438, 208.8813629
1: -115.7975693, 385.7521057, -109.9310150, 367.1177673, -482.9152832, 495.6830444
2: -167.5631256, 337.7823181, -158.4000397, 322.8183594, -490.3814697, 496.1823730
3: -98.9471817, 406.9516296, -93.8630524, 386.8898010, -485.8369751, 500.8146973
4: -154.5105133, 292.7888794, -146.2857971, 279.6833496, -434.1938171, 439.0746765

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A1_A2_B1

### Relational analysis result of NS_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7267766
time: 0.85 seconds

## Relational analysis of NS_B1_B2_B1_A1_A2_B2

### Relational analysis result of NS_B1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7275000
time: 0.93 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -43.8816643, 168.8896484, -41.4450569, 159.4190521, -203.3007202, 210.3347015
1: -116.6261902, 388.2770386, -109.9310150, 367.1177673, -483.7439270, 498.2079773
2: -168.2041016, 340.8746643, -158.4000397, 322.8183594, -491.0224609, 499.2747192
3: -99.5314713, 409.6002502, -93.8630524, 386.8898010, -486.4212341, 503.4633179
4: -155.1987457, 295.2785950, -146.2857971, 279.6833496, -434.8820190, 441.5643921

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B1_A2_A1_A1

### Relational analysis result of NS_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275097
time: 0.69 seconds

## Relational analysis of NS_B1_B2_B1_A2_A1_A2

### Relational analysis result of NS_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275113
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -45.7875977, 176.0248413, -41.4450569, 159.4190521, -205.2066498, 217.4698944
1: -122.0026855, 405.1712952, -109.9310150, 367.1177673, -489.1204529, 515.1022949
2: -176.1266174, 354.7684937, -158.4000397, 322.8183594, -498.9449768, 513.1685181
3: -104.1263733, 427.8629456, -93.8630524, 386.8898010, -491.0161743, 521.7260132
4: -162.4115448, 307.3793640, -146.2857971, 279.6833496, -442.0949097, 453.6651611

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B1_A2_A2_A1

### Relational analysis result of NS_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275097
time: 0.82 seconds

## Relational analysis of NS_B1_B2_B1_A2_A2_A2

### Relational analysis result of NS_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275113
time: 0.84 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -42.1634254, 162.1604309, -43.2720642, 166.5788879, -208.7423096, 205.4324493
1: -111.8679581, 373.4281616, -114.9974594, 382.9272156, -494.7951660, 488.4255981
2: -161.2144623, 328.1668701, -165.8288574, 336.3431091, -497.5575562, 493.9957275
3: -95.5358963, 393.7258301, -98.1179657, 403.8304749, -499.3663635, 491.8438110
4: -148.8794556, 284.3475037, -153.0210419, 291.3052368, -440.1846619, 437.3685303

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A1_A1_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274401, upper bound: 187.7274626
time: 0.77 seconds

## Relational analysis of NS_B1_B2_B2_A1_A1_A2

### Relational analysis result of NS_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -43.5457916, 167.4363098, -43.2720642, 166.5788879, -210.1246796, 210.7083588
1: -115.7975693, 385.7521057, -114.9974594, 382.9272156, -498.7247620, 500.7495422
2: -167.5631256, 337.7823181, -165.8288574, 336.3431091, -503.9062500, 503.6111145
3: -98.9471817, 406.9516296, -98.1179657, 403.8304749, -502.7776489, 505.0695801
4: -154.5105133, 292.7888794, -153.0210419, 291.3052368, -445.8157349, 445.8099060

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A1_A2_A1

### Relational analysis result of NS_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274401, upper bound: 187.7274626
time: 0.80 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2_A2

### Relational analysis result of NS_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
time: 0.78 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -43.8816643, 168.8896484, -43.2720642, 166.5788879, -210.4605560, 212.1616821
1: -116.6261902, 388.2770386, -114.9974594, 382.9272156, -499.5533752, 503.2745056
2: -168.2041016, 340.8746643, -165.8288574, 336.3431091, -504.5472107, 506.7035217
3: -99.5314713, 409.6002502, -98.1179657, 403.8304749, -503.3619080, 507.7182007
4: -155.1987457, 295.2785950, -153.0210419, 291.3052368, -446.5038757, 448.2995911

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7274890
time: 0.74 seconds

## Relational analysis of NS_B1_B2_B2_A2_A1_A2

### Relational analysis result of NS_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275293, upper bound: 187.7275273
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -45.7875977, 176.0248413, -43.2720642, 166.5788879, -212.3664856, 219.2969055
1: -122.0026855, 405.1712952, -114.9974594, 382.9272156, -504.9299011, 520.1687622
2: -176.1266174, 354.7684937, -165.8288574, 336.3431091, -512.4697266, 520.5973511
3: -104.1263733, 427.8629456, -98.1179657, 403.8304749, -507.9568481, 525.9808960
4: -162.4115448, 307.3793640, -153.0210419, 291.3052368, -453.7167664, 460.4003906

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7274892
time: 0.71 seconds

## Relational analysis of NS_B1_B2_B2_A2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7275273
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.7992821, 148.9391174, -40.8932076, 157.0306091, -195.8298950, 189.8323212
1: -103.0423737, 342.4579773, -108.4743881, 361.2658386, -464.3081055, 450.9323730
2: -150.0254669, 300.8915100, -157.4253845, 317.5674133, -467.5928955, 458.3168640
3: -88.1236801, 361.4491577, -92.7406845, 380.8136902, -468.9373474, 454.1898499
4: -138.1436920, 260.8076477, -145.1132507, 275.1289673, -413.2726440, 405.9208069

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7244345, upper bound: 187.7248810
time: 0.71 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252118, upper bound: 187.7257867
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.6596642, 156.2635956, -41.6719589, 160.2366028, -200.8962708, 197.9355469
1: -108.0037231, 359.8626404, -110.7073975, 369.5130920, -477.5168152, 470.5700378
2: -156.4745483, 315.5716858, -160.1823425, 323.4183655, -479.8928528, 475.7540283
3: -92.3574066, 379.9284973, -94.6187897, 389.8112183, -482.1686401, 474.5473022
4: -144.2637024, 273.6245422, -147.6698608, 280.3576660, -424.6213379, 421.2943420

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7243321, upper bound: 187.7249082
time: 0.70 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252404, upper bound: 187.7256589
time: 0.70 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252404, upper bound: 187.7261348
time: 0.68 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -40.6696739, 156.2474823, -40.8932076, 157.0306091, -197.7002869, 197.1406860
1: -108.3022766, 358.9060364, -108.4743881, 361.2658386, -469.5680542, 467.3804321
2: -157.5028839, 314.8746033, -157.4253845, 317.5674133, -475.0702515, 472.2998962
3: -92.4845123, 379.0491638, -92.7406845, 380.8136902, -473.2982178, 471.7898560
4: -144.9008636, 272.7808228, -145.1132507, 275.1289673, -420.0298462, 417.8940735

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7244816, upper bound: 187.7249413
time: 0.68 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252846, upper bound: 187.7258621
time: 0.69 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -42.3749924, 162.8811340, -41.6719589, 160.2366028, -202.6116028, 204.5530853
1: -112.7436066, 374.6596985, -110.7073975, 369.5130920, -482.2567139, 485.3670959
2: -163.2665863, 328.3375549, -160.1823425, 323.4183655, -486.6848145, 488.5198669
3: -96.2943954, 395.6973877, -94.6187897, 389.8112183, -486.1056213, 490.3161621
4: -150.3695374, 284.5310974, -147.6698608, 280.3576660, -430.7272034, 432.2009277

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253927, upper bound: 187.7259711
time: 0.79 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254972, upper bound: 187.7261578
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.7992821, 148.9391174, -43.2606735, 166.1111145, -204.9104004, 192.1997681
1: -103.0423737, 342.4579773, -115.0490875, 382.2565613, -485.2988892, 457.5070801
2: -150.0254669, 300.8915100, -166.2665558, 335.5995483, -485.6250000, 467.1580811
3: -88.1236801, 361.4491577, -98.2026596, 403.2772217, -491.4008484, 459.6518250
4: -138.1436920, 260.8076477, -153.2707977, 290.6332397, -428.7769165, 414.0783386

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250403, upper bound: 187.7255916
time: 0.72 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253317, upper bound: 187.7258259
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -40.6596642, 156.2635956, -44.0509834, 169.4802704, -210.1399231, 200.3145752
1: -108.0037231, 359.8626404, -117.3094101, 390.3150024, -498.3187256, 477.1720581
2: -156.4745483, 315.5716858, -169.3454132, 341.3682251, -497.8427734, 484.9171143
3: -92.3574066, 379.9284973, -100.1119080, 412.1177979, -504.4751892, 480.0404053
4: -144.2637024, 273.6245422, -156.1042328, 295.7691650, -440.0327759, 429.7286682

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269273, upper bound: 187.7271933
time: 1.01 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268982, upper bound: 187.7271756
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.6696739, 156.2474823, -43.2606735, 166.1111145, -206.7807922, 199.5081329
1: -108.3022766, 358.9060364, -115.0490875, 382.2565613, -490.5588074, 473.9551392
2: -157.5028839, 314.8746033, -166.2665558, 335.5995483, -493.1024170, 481.1411438
3: -92.4845123, 379.0491638, -98.2026596, 403.2772217, -495.7617188, 477.2518311
4: -144.9008636, 272.7808228, -153.2707977, 290.6332397, -435.5340881, 426.0516052

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7244222, upper bound: 187.7246083
time: 0.74 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255817, upper bound: 187.7260741
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.3749924, 162.8811340, -44.0509834, 169.4802704, -211.8552551, 206.9321136
1: -112.7436066, 374.6596985, -117.3094101, 390.3150024, -503.0585938, 491.9691162
2: -163.2665863, 328.3375549, -169.3454132, 341.3682251, -504.6347351, 497.6828613
3: -96.2943954, 395.6973877, -100.1119080, 412.1177979, -508.4121704, 495.8092957
4: -150.3695374, 284.5310974, -156.1042328, 295.7691650, -446.1387024, 440.6353149

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -44.3195229, 169.9568787, -41.9552193, 161.2417297, -205.5612488, 211.9120941
1: -117.8857803, 390.8894043, -111.5608521, 371.6982422, -489.5840149, 502.4502563
2: -171.6435699, 342.9281006, -161.9490662, 325.0089722, -496.6524963, 504.8771057
3: -100.8268814, 412.9217834, -95.4081497, 392.4071960, -493.2340698, 508.3299255
4: -158.0062561, 297.1517639, -149.1902313, 281.8039856, -439.8102417, 446.3419800

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7252404
time: 0.66 seconds

## Relational analysis of NS_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7254180
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -46.9658508, 180.0115814, -41.9552193, 161.2417297, -208.2075806, 221.9667969
1: -125.1928253, 413.9126282, -111.5608521, 371.6982422, -496.8910522, 525.4734497
2: -181.7768250, 362.9929810, -161.9490662, 325.0089722, -506.7857971, 524.9419556
3: -106.9251785, 437.3711243, -95.4081497, 392.4071960, -499.3323669, 532.7791748
4: -167.3488464, 314.4036560, -149.1902313, 281.8039856, -449.1528320, 463.5938721

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7259899
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261348, upper bound: 187.7269273
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -44.3195229, 169.9568787, -44.2351913, 170.1514893, -214.4710083, 214.1920776
1: -117.8857803, 390.8894043, -117.9425812, 391.9692383, -509.8550110, 508.8319092
2: -171.6435699, 342.9281006, -170.6576538, 342.3324890, -513.9760132, 513.5857544
3: -100.8268814, 412.9217834, -100.6752853, 414.0962524, -514.9231567, 513.5970459
4: -158.0062561, 297.1517639, -157.1894226, 296.6888428, -454.6950989, 454.3411865

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251557, upper bound: 187.7250653
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261335, upper bound: 187.7254972
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -46.9658508, 180.0115814, -44.2351913, 170.1514893, -217.1173401, 224.2467651
1: -125.1928253, 413.9126282, -117.9425812, 391.9692383, -517.1620483, 531.8551025
2: -181.7768250, 362.9929810, -170.6576538, 342.3324890, -524.1093140, 533.6506348
3: -106.9251785, 437.3711243, -100.6752853, 414.0962524, -521.0213013, 538.0463867
4: -167.3488464, 314.4036560, -157.1894226, 296.6888428, -464.0376892, 471.5930786

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251557, upper bound: 187.7260083
time: 0.86 seconds

## Relational analysis of NS_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261335, upper bound: 187.7277065
time: 0.90 seconds

## BFS NS instance: NS_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -44.5447235, 171.0336304, -44.5086441, 170.4142761, -214.9589996, 215.5422516
1: -118.3685379, 393.6081543, -118.7401810, 391.3509521, -509.7194824, 512.3482666
2: -170.7965088, 345.7730408, -172.3686218, 343.1094971, -513.9060059, 518.1416626
3: -101.0960083, 415.3291016, -101.5092850, 414.5434265, -515.6394043, 516.8383789
4: -157.7216492, 299.4778442, -158.6867676, 297.3498840, -455.0715332, 458.1646118

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B2_B1_B1_B1

### Relational analysis result of NS_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267248, upper bound: 187.7267240
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B2_B1_B1_B2

### Relational analysis result of NS_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271895, upper bound: 187.7268837
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -44.6460152, 171.4276123, -44.6680756, 171.1531830, -215.7991791, 216.0956879
1: -118.6471863, 394.4845886, -119.1840820, 393.1224976, -511.7695923, 513.6686401
2: -171.2498932, 346.5161438, -172.9107666, 344.4446411, -515.6945190, 519.4268799
3: -101.3358688, 416.2548218, -101.8714828, 416.3262329, -517.6621094, 518.1262817
4: -158.1157837, 300.1214294, -159.2129822, 298.5102234, -456.6260071, 459.3344116

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B2_B1_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266295, upper bound: 187.7266295
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B2_B1_B2_B2

### Relational analysis result of NS_B2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266295, upper bound: 187.7267580
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -44.3024635, 169.9454956, -46.4324760, 178.0702820, -222.3727264, 216.3779755
1: -117.8231430, 390.7470398, -123.8275986, 409.3616028, -527.1847534, 514.5746460
2: -171.3757019, 343.0940552, -180.5077362, 358.4240417, -529.7997437, 523.6018066
3: -100.7317963, 412.5205688, -105.9154358, 432.7742004, -533.5059204, 518.4360352
4: -157.8114166, 297.2198792, -166.0286713, 310.6317139, -468.4431152, 463.2485046

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263407, upper bound: 187.7263407
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263407, upper bound: 187.7267210
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -44.6142464, 171.1967010, -49.2993011, 189.1831055, -233.7973328, 220.4959717
1: -118.6849976, 393.4432678, -131.7699127, 435.2558899, -553.9408569, 525.2131958
2: -172.3767853, 345.7039490, -191.0642242, 380.4259338, -552.8027344, 536.7681885
3: -101.4222183, 415.2906494, -112.5079803, 460.4155884, -561.8377686, 527.7985229
4: -158.7782135, 299.4356079, -175.8488922, 329.6290588, -488.4072876, 475.2844849

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267210, upper bound: 187.7266764
time: 0.72 seconds

## Relational analysis of NS_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267210, upper bound: 187.7273049
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.68 seconds
NS_B1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
NS_B1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275116
NS_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275118
NS_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275116
NS_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275390
NS_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275164, upper bound: 187.7275293
NS_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275390
NS_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275164, upper bound: 187.7275293
NS_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
NS_B1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274895, upper bound: 187.7275407
NS_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274942, upper bound: 187.7275769
NS_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274895, upper bound: 187.7275407
NS_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275310, upper bound: 187.7275629
NS_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275311, upper bound: 187.7275444
NS_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275310, upper bound: 187.7275631
NS_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274668, upper bound: 187.7275444
NS_B1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7259318, upper bound: 187.7260912
NS_B1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7266851, upper bound: 187.7266318
NS_B1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7267766
NS_B1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7275000
NS_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275097
NS_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275113
NS_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275097
NS_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275113
NS_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274401, upper bound: 187.7274626
NS_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
NS_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274401, upper bound: 187.7274626
NS_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274794, upper bound: 187.7275210
NS_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7274890
NS_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7275293, upper bound: 187.7275273
NS_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7274892
NS_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274979, upper bound: 187.7275273
NS_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7244345, upper bound: 187.7248810
NS_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7252118, upper bound: 187.7257867
NS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7252404, upper bound: 187.7256589
NS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7252404, upper bound: 187.7261348
NS_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7244816, upper bound: 187.7249413
NS_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7252846, upper bound: 187.7258621
NS_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7253927, upper bound: 187.7259711
NS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7254972, upper bound: 187.7261578
NS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7250403, upper bound: 187.7255916
NS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7253317, upper bound: 187.7258259
NS_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7269273, upper bound: 187.7271933
NS_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7268982, upper bound: 187.7271756
NS_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7244222, upper bound: 187.7246083
NS_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7255817, upper bound: 187.7260741
NS_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
NS_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
NS_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7252404
NS_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7254180
NS_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7259899
NS_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7261348, upper bound: 187.7269273
NS_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7251557, upper bound: 187.7250653
NS_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7261335, upper bound: 187.7254972
NS_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7251557, upper bound: 187.7260083
NS_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7261335, upper bound: 187.7277065
NS_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7267248, upper bound: 187.7267240
NS_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7271895, upper bound: 187.7268837
NS_B2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7266295, upper bound: 187.7266295
NS_B2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7266295, upper bound: 187.7267580
NS_B2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7263407, upper bound: 187.7263407
NS_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7263407, upper bound: 187.7267210
NS_B2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7267210, upper bound: 187.7266764
NS_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -187.7267210, upper bound: 187.7273049

## BFS NS instance: NS_B1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3620186, 159.1649475, -38.8011246, 148.9701843, -190.3321991, 197.9660339
1: -109.6215668, 367.0632629, -102.8347092, 342.1187134, -451.7402954, 469.8978882
2: -157.5258636, 322.2335510, -148.4420166, 301.9038391, -459.4296875, 470.6755676
3: -93.6532593, 386.9813538, -87.8598328, 361.0752563, -454.7285156, 474.8411865
4: -145.6114044, 279.2987061, -137.0674591, 261.5943909, -407.2057800, 416.3661499

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271608, upper bound: 187.7271510
time: 1.02 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265234, upper bound: 187.7259046
time: 0.95 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -41.4464417, 159.4935913, -39.0045471, 149.7804413, -191.2268372, 198.4981384
1: -109.8446350, 367.8159180, -103.3794327, 344.0171814, -453.8617859, 471.1953430
2: -157.8076172, 322.9380798, -148.9832916, 303.7160034, -461.5235901, 471.9213562
3: -93.8363190, 387.7505493, -88.2883606, 363.0966187, -456.9329224, 476.0389099
4: -145.8804016, 279.9027100, -137.6230774, 263.1687927, -409.0491943, 417.5257874

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271140, upper bound: 187.7270869
time: 0.72 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265741, upper bound: 187.7259527
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -42.2328682, 162.5878296, -38.8011246, 148.9701843, -191.2030334, 201.3889008
1: -112.1053696, 375.2148743, -102.8347092, 342.1187134, -454.2240601, 478.0495300
2: -161.0789948, 328.5521851, -148.4420166, 301.9038391, -462.9828491, 476.9942017
3: -95.7183380, 395.6139221, -87.8598328, 361.0752563, -456.7935791, 483.4737549
4: -148.8247223, 284.7714539, -137.0674591, 261.5943909, -410.4190674, 421.8388977

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272935, upper bound: 187.7273043
time: 0.68 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270219, upper bound: 187.7268555
time: 0.78 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270219, upper bound: 187.7271289
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -42.3334045, 162.9764709, -39.0045471, 149.7804413, -192.1138000, 201.9810181
1: -112.4021759, 376.0254517, -103.3794327, 344.0171814, -456.4193420, 479.4048767
2: -161.5631409, 329.2550049, -148.9832916, 303.7160034, -465.2791138, 478.2382812
3: -95.9755936, 396.5129395, -88.2883606, 363.0966187, -459.0721741, 484.8013000
4: -149.2608795, 285.3800049, -137.6230774, 263.1687927, -412.4296570, 423.0030518

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273522, upper bound: 187.7273650
time: 0.82 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270644, upper bound: 187.7268810
time: 0.69 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272794, upper bound: 187.7271250
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -42.8960609, 165.1033630, -38.8011246, 148.9701843, -191.8662415, 203.9044800
1: -113.8371506, 379.9886780, -102.8347092, 342.1187134, -455.9558411, 482.8233643
2: -163.4312744, 333.5563965, -148.4420166, 301.9038391, -465.3351135, 481.9984131
3: -97.1632614, 400.8609619, -87.8598328, 361.0752563, -458.2384949, 488.7207947
4: -151.0240021, 289.0090332, -137.0674591, 261.5943909, -412.6184082, 426.0764771

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275388
time: 1.19 seconds

## Relational analysis of NS_B1_B1_B1_A2_A1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275390
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -42.8825493, 165.0646820, -39.0045471, 149.7804413, -192.6629486, 204.0692291
1: -113.8079758, 379.8580017, -103.3794327, 344.0171814, -457.8251343, 483.2374268
2: -163.4058990, 333.4668274, -148.9832916, 303.7160034, -467.1218567, 482.4501038
3: -97.1329117, 400.7079468, -88.2883606, 363.0966187, -460.2294922, 488.9962769
4: -150.9902496, 288.9254456, -137.6230774, 263.1687927, -414.1590576, 426.5484924

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275164, upper bound: 187.7275276
time: 0.74 seconds

## Relational analysis of NS_B1_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275160, upper bound: 187.7275293
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -44.3103409, 170.3726501, -38.8011246, 148.9701843, -193.2804718, 209.1737518
1: -117.8133545, 392.4183655, -102.8347092, 342.1187134, -459.9320374, 495.2530212
2: -169.0890808, 344.2160950, -148.4420166, 301.9038391, -470.9928589, 492.6581116
3: -100.5117264, 414.0545654, -87.8598328, 361.0752563, -461.5869141, 501.9143982
4: -156.2083740, 298.2258606, -137.0674591, 261.5943909, -417.8027649, 435.2933044

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_B1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274927, upper bound: 187.7273240
time: 0.79 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274894, upper bound: 187.7274749
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -44.3734055, 170.6127014, -39.0045471, 149.7804413, -194.1537933, 209.6172485
1: -118.0114822, 392.9021301, -103.3794327, 344.0171814, -462.0286255, 496.2815552
2: -169.4035187, 344.6582642, -148.9832916, 303.7160034, -473.1195068, 493.6415405
3: -100.6756058, 414.6115723, -88.2883606, 363.0966187, -463.7722168, 502.8999329
4: -156.4834137, 298.6102600, -137.6230774, 263.1687927, -419.6522217, 436.2333069

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275411, upper bound: 187.7275276
time: 0.71 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275411, upper bound: 187.7275293
time: 0.89 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3620186, 159.1649475, -39.6060181, 152.0836639, -193.4456787, 198.7709656
1: -109.6215668, 367.0632629, -105.1983109, 349.5744324, -459.1959839, 472.2615662
2: -157.5258636, 322.2335510, -151.9415588, 307.0794373, -464.6052856, 474.1751099
3: -93.6532593, 386.9813538, -89.8730698, 369.2666931, -462.9199524, 476.8544312
4: -145.6114044, 279.2987061, -140.1973877, 266.0708008, -411.6821899, 419.4960938

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_B1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273264, upper bound: 187.7275771
time: 0.94 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_B2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274838, upper bound: 187.7275806
time: 0.88 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -41.4464417, 159.4935913, -39.8156357, 152.9652100, -194.4116364, 199.3092346
1: -109.8446350, 367.8159180, -105.7431107, 351.6112976, -461.4559021, 473.5590210
2: -157.8076172, 322.9380798, -152.5380707, 308.9847412, -466.7923584, 475.4760742
3: -93.8363190, 387.7505493, -90.3056030, 371.3290710, -465.1654053, 478.0561523
4: -145.8804016, 279.9027100, -140.7744141, 267.7158203, -413.5962219, 420.6771240

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274987, upper bound: 187.7275412
time: 0.77 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275209, upper bound: 187.7275509
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -42.2328682, 162.5878296, -39.6060181, 152.0836639, -194.3165283, 202.1938477
1: -112.1053696, 375.2148743, -105.1983109, 349.5744324, -461.6797791, 480.4131775
2: -161.0789948, 328.5521851, -151.9415588, 307.0794373, -468.1584473, 480.4937439
3: -95.7183380, 395.6139221, -89.8730698, 369.2666931, -464.9850464, 485.4869995
4: -148.8247223, 284.7714539, -140.1973877, 266.0708008, -414.8954773, 424.9688416

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270928, upper bound: 187.7269260
time: 0.92 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270928, upper bound: 187.7272020
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -42.3334045, 162.9764709, -39.8156357, 152.9652100, -195.2985992, 202.7921143
1: -112.4021759, 376.0254517, -105.7431107, 351.6112976, -464.0134583, 481.7685547
2: -161.5631409, 329.2550049, -152.5380707, 308.9847412, -470.5478516, 481.7930603
3: -95.9755936, 396.5129395, -90.3056030, 371.3290710, -467.3045959, 486.8185425
4: -149.2608795, 285.3800049, -140.7744141, 267.7158203, -416.9766846, 426.1543884

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270902, upper bound: 187.7269315
time: 1.11 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270902, upper bound: 187.7271587
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -42.8960609, 165.1033630, -39.6060181, 152.0836639, -194.9797211, 204.7093811
1: -113.8371506, 379.9886780, -105.1983109, 349.5744324, -463.4115601, 485.1869812
2: -163.4312744, 333.5563965, -151.9415588, 307.0794373, -470.5106812, 485.4979553
3: -97.1632614, 400.8609619, -89.8730698, 369.2666931, -466.4299622, 490.7340393
4: -151.0240021, 289.0090332, -140.1973877, 266.0708008, -417.0947876, 429.2064209

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_B1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273474, upper bound: 187.7274541
time: 0.81 seconds

## Relational analysis of NS_B1_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_B1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274513, upper bound: 187.7275287
time: 0.88 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -42.8825493, 165.0646820, -39.8156357, 152.9652100, -195.8477478, 204.8803101
1: -113.8079758, 379.8580017, -105.7431107, 351.6112976, -465.4192200, 485.6011047
2: -163.4058990, 333.4668274, -152.5380707, 308.9847412, -472.3906250, 486.0048523
3: -97.1329117, 400.7079468, -90.3056030, 371.3290710, -468.4619446, 491.0135498
4: -150.9902496, 288.9254456, -140.7744141, 267.7158203, -418.7060547, 429.6998291

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275311, upper bound: 187.7275444
time: 0.83 seconds

## Relational analysis of NS_B1_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275311, upper bound: 187.7275444
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -44.3103409, 170.3726501, -39.6060181, 152.0836639, -196.3939667, 209.9786682
1: -117.8133545, 392.4183655, -105.1983109, 349.5744324, -467.3877563, 497.6166687
2: -169.0890808, 344.2160950, -151.9415588, 307.0794373, -476.1684265, 496.1576233
3: -100.5117264, 414.0545654, -89.8730698, 369.2666931, -469.7783508, 503.9276428
4: -156.2083740, 298.2258606, -140.1973877, 266.0708008, -422.2791138, 438.4232483

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_B1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275195, upper bound: 187.7273960
time: 1.08 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275155, upper bound: 187.7274992
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -44.3734055, 170.6127014, -39.8156357, 152.9652100, -197.3385925, 210.4283447
1: -118.0114822, 392.9021301, -105.7431107, 351.6112976, -469.6227417, 498.6452332
2: -169.4035187, 344.6582642, -152.5380707, 308.9847412, -478.3882446, 497.1962891
3: -100.6756058, 414.6115723, -90.3056030, 371.3290710, -472.0046692, 504.9171753
4: -156.4834137, 298.6102600, -140.7744141, 267.7158203, -424.1992188, 439.3846436

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274945, upper bound: 187.7273437
time: 0.68 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275081, upper bound: 187.7274891
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -40.0776558, 153.9933014, -39.7641373, 152.8382874, -192.9159393, 193.7574463
1: -106.2385864, 354.1017456, -105.1517792, 351.6053162, -457.8438721, 459.2534485
2: -153.5326996, 312.0913086, -151.5733490, 310.5929565, -464.1256409, 463.6646729
3: -90.7108459, 373.1076355, -89.7837219, 369.9735413, -460.6843872, 462.8913574
4: -141.6857300, 270.3530884, -140.0100403, 268.8871765, -410.5729065, 410.3631287

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7257281, upper bound: 187.7255771
time: 0.81 seconds

## Relational analysis of NS_B1_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259815, upper bound: 187.7261556
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -41.6830444, 160.3358917, -40.5732689, 156.1282196, -197.8112640, 200.9091644
1: -110.5434189, 369.2758789, -107.5094833, 359.6452026, -470.1886292, 476.7853699
2: -159.1257629, 324.6295471, -154.5388641, 316.4661255, -475.5918884, 479.1683960
3: -94.3893661, 389.2873230, -91.7661438, 378.8974304, -473.2867432, 481.0534668
4: -146.9999542, 281.2755432, -142.8030701, 274.1586304, -421.1585693, 424.0786133

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267288, upper bound: 187.7267288
time: 0.96 seconds

## Relational analysis of NS_B1_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267288, upper bound: 187.7267288
time: 0.69 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -42.9079628, 164.9647675, -42.7580299, 164.6201477, -207.5280914, 207.7227936
1: -114.1230392, 379.8162231, -113.4090729, 378.6454468, -492.7684326, 493.2252808
2: -165.2532349, 332.7807312, -162.9000702, 333.1796570, -498.4328918, 495.6807556
3: -97.5124359, 400.5673828, -96.7892456, 398.9320374, -496.4444580, 497.3565979
4: -152.3576050, 288.4347534, -150.6042633, 288.5145569, -440.8721619, 439.0390015

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264956, upper bound: 187.7263156
time: 1.09 seconds

## Relational analysis of NS_B1_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264956, upper bound: 187.7267766
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -43.4118309, 166.9189301, -40.9841919, 157.6393738, -201.0511932, 207.9031219
1: -115.4275742, 384.5754700, -108.6698837, 363.0599060, -478.4874573, 493.2453613
2: -167.0321503, 336.7676086, -156.5972900, 319.3187561, -486.3508911, 493.3648987
3: -98.6307755, 405.6860962, -92.7865982, 382.5552979, -481.1860046, 498.4726868
4: -154.0188446, 291.9101562, -144.6162415, 276.6414490, -430.6602783, 436.5263977

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268059, upper bound: 187.7265785
time: 0.74 seconds

## Relational analysis of NS_B1_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268059, upper bound: 187.7275000
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -40.3673172, 155.0816498, -41.4450569, 159.4190521, -199.7863770, 196.5267029
1: -107.2326202, 356.4651489, -109.9310150, 367.1177673, -474.3503113, 466.3961182
2: -154.6658783, 313.1633606, -158.4000397, 322.8183594, -477.4842529, 471.5634155
3: -91.5851974, 376.5257263, -93.8630524, 386.8898010, -478.4750061, 470.3887939
4: -142.7607880, 271.3407288, -146.2857971, 279.6833496, -422.4440918, 417.6265259

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A2_A1_A1_B1

### Relational analysis result of NS_B1_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274802, upper bound: 187.7274756
time: 0.73 seconds

## Relational analysis of NS_B1_B2_B1_A2_A1_A1_B2

### Relational analysis result of NS_B1_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274802, upper bound: 187.7275097
time: 0.73 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -43.2720642, 166.5788879, -41.4450569, 159.4190521, -202.6911011, 208.0239410
1: -114.9974594, 382.9272156, -109.9310150, 367.1177673, -482.1152039, 492.8581543
2: -165.8288574, 336.3431091, -158.4000397, 322.8183594, -488.6472168, 494.7431641
3: -98.1179657, 403.8304749, -93.8630524, 386.8898010, -485.0077515, 497.6935425
4: -153.0210419, 291.3052368, -146.2857971, 279.6833496, -432.7044067, 437.5910339

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A2_A1_A2_B1

### Relational analysis result of NS_B1_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274802, upper bound: 187.7274779
time: 0.73 seconds

## Relational analysis of NS_B1_B2_B1_A2_A1_A2_B2

### Relational analysis result of NS_B1_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275162, upper bound: 187.7275113
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -41.7060051, 160.4754028, -41.4450569, 159.4190521, -201.1250610, 201.9204559
1: -111.1794815, 369.3103333, -109.9310150, 367.1177673, -478.2972107, 479.2413025
2: -160.0488586, 323.1542664, -158.4000397, 322.8183594, -482.8672180, 481.5543213
3: -94.7906570, 390.2597961, -93.8630524, 386.8898010, -481.6804504, 484.1228638
4: -147.6875305, 279.9117126, -146.2857971, 279.6833496, -427.3708191, 426.1975098

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A2_A2_A1_B1

### Relational analysis result of NS_B1_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275030, upper bound: 187.7274755
time: 1.06 seconds

## Relational analysis of NS_B1_B2_B1_A2_A2_A1_B2

### Relational analysis result of NS_B1_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275402, upper bound: 187.7275097
time: 0.87 seconds

## BFS NS instance: NS_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -45.1959229, 173.7508850, -41.4450569, 159.4190521, -204.6149750, 215.1959381
1: -120.3991394, 399.9425049, -109.9310150, 367.1177673, -487.5168457, 509.8734436
2: -173.8061218, 350.3166504, -158.4000397, 322.8183594, -496.6244812, 508.7166748
3: -102.7521133, 422.2483521, -93.8630524, 386.8898010, -489.6419067, 516.1113892
4: -160.2844543, 303.4986877, -146.2857971, 279.6833496, -439.9678040, 449.7844849

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_B1_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275030, upper bound: 187.7274779
time: 0.80 seconds

## Relational analysis of NS_B1_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_B1_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275402, upper bound: 187.7275113
time: 0.82 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -41.1318741, 158.0942993, -42.9917183, 165.4942169, -206.6260529, 201.0860138
1: -109.0894699, 363.7544556, -114.2268143, 380.4856873, -489.5751343, 477.9812622
2: -157.3844910, 320.0945740, -164.7067108, 334.2024231, -491.5869141, 484.8012695
3: -93.1606827, 383.4754639, -97.4634171, 401.2392578, -494.3999329, 480.9388733
4: -145.2976837, 277.3774414, -151.9860229, 289.4570923, -434.7546387, 429.3634644

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A1_A1_A1_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274647, upper bound: 187.7274802
time: 0.85 seconds

## Relational analysis of NS_B1_B2_B2_A1_A1_A1_A2

### Relational analysis result of NS_B1_B2_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274647, upper bound: 187.7274802
time: 0.88 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -41.6745262, 160.2696686, -43.0150833, 165.5869598, -207.2614899, 203.2847595
1: -110.5520325, 369.0783386, -114.3105240, 380.6470642, -491.1990356, 483.3888550
2: -159.3383942, 324.4458008, -164.8565979, 334.3685913, -493.7069702, 489.3023987
3: -94.4064102, 389.0897217, -97.5303955, 401.4034119, -495.8098145, 486.6200562
4: -147.1284332, 281.1221008, -152.1155243, 289.5935364, -436.7219849, 433.2376099

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A1_A1_A2_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275097, upper bound: 187.7275162
time: 0.81 seconds

## Relational analysis of NS_B1_B2_B2_A1_A1_A2_A2

### Relational analysis result of NS_B1_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274756, upper bound: 187.7275274
time: 0.73 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -42.5999947, 163.6053314, -42.9917183, 165.4942169, -208.0941925, 206.5970459
1: -113.3026581, 376.7072754, -114.2268143, 380.4856873, -493.7883301, 490.9340820
2: -164.3771362, 330.1240234, -164.7067108, 334.2024231, -498.5795593, 494.8307495
3: -96.8471375, 397.5323181, -97.4634171, 401.2392578, -498.0863953, 494.9957275
4: -151.4463348, 286.1375122, -151.9860229, 289.4570923, -440.9032898, 438.1235352

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B1

### Relational analysis result of NS_B1_B2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273921, upper bound: 187.7274297
time: 0.71 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2_A1_B2

### Relational analysis result of NS_B1_B2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274043, upper bound: 187.7274626
time: 1.17 seconds

## BFS NS instance: NS_B1_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -43.0321617, 165.4560394, -43.0150833, 165.5869598, -208.6191254, 208.4711151
1: -114.4360809, 381.1740417, -114.3105240, 380.6470642, -495.0830688, 495.4845581
2: -165.6715851, 333.7866821, -164.8565979, 334.3685913, -500.0401611, 498.6431885
3: -97.7883759, 402.1218872, -97.5303955, 401.4034119, -499.1917725, 499.6522522
4: -152.7373810, 289.3293152, -152.1155243, 289.5935364, -442.3309326, 441.4448242

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270511, upper bound: 187.7269241
time: 0.83 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271501, upper bound: 187.7271280
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -43.0686722, 165.6482391, -42.9917183, 165.4942169, -208.5628967, 208.6399536
1: -114.4943237, 380.7650757, -114.2268143, 380.4856873, -494.9800110, 494.9918823
2: -165.4995880, 334.1690063, -164.7067108, 334.2024231, -499.7020264, 498.8756714
3: -97.7409210, 401.8094788, -97.4634171, 401.2392578, -498.9801636, 499.2728882
4: -152.5944672, 289.4846802, -151.9860229, 289.4570923, -442.0514526, 441.4706726

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A2_A1_A1_A1

### Relational analysis result of NS_B1_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274939, upper bound: 187.7274890
time: 0.70 seconds

## Relational analysis of NS_B1_B2_B2_A2_A1_A1_A2

### Relational analysis result of NS_B1_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274939, upper bound: 187.7274890
time: 0.78 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -43.2489586, 166.4453888, -43.0150833, 165.5869598, -208.8358917, 209.4604797
1: -114.9294891, 382.6718445, -114.3105240, 380.6470642, -495.5764771, 496.9823608
2: -165.7885742, 336.0371704, -164.8565979, 334.3685913, -500.1571655, 500.8937683
3: -98.0773926, 403.6149597, -97.5303955, 401.4034119, -499.4808044, 501.1453247
4: -152.9508057, 291.0813293, -152.1155243, 289.5935364, -442.5443420, 443.1968384

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A2_A1_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274647, upper bound: 187.7275273
time: 1.14 seconds

## Relational analysis of NS_B1_B2_B2_A2_A1_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275273, upper bound: 187.7275273
time: 0.65 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -45.0200348, 173.0914307, -42.9917183, 165.4942169, -210.5142212, 216.0831451
1: -120.0355606, 398.4222412, -114.2268143, 380.4856873, -500.5212097, 512.6490479
2: -173.5943451, 348.4528503, -164.7067108, 334.2024231, -507.7967529, 513.1595459
3: -102.4513321, 420.8719482, -97.4634171, 401.2392578, -503.6905823, 518.3353271
4: -159.9654388, 301.9254761, -151.9860229, 289.4570923, -449.4225159, 453.9114990

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A2_A2_A1_A1

### Relational analysis result of NS_B1_B2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275500, upper bound: 187.7274892
time: 0.76 seconds

## Relational analysis of NS_B1_B2_B2_A2_A2_A1_A2

### Relational analysis result of NS_B1_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275500, upper bound: 187.7274892
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -45.1799355, 173.6825867, -43.0150833, 165.5869598, -210.7668762, 216.6976624
1: -120.3770142, 399.7647400, -114.3105240, 380.6470642, -501.0240479, 514.0751343
2: -173.8594360, 350.0899048, -164.8565979, 334.3685913, -508.2280273, 514.9465332
3: -102.7377853, 422.0840759, -97.5303955, 401.4034119, -504.1412048, 519.6145020
4: -160.2948456, 303.3117676, -152.1155243, 289.5935364, -449.8883667, 455.4273071

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272014, upper bound: 187.7268509
time: 0.80 seconds

## Relational analysis of NS_B1_B2_B2_A2_A2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272411, upper bound: 187.7271884
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -38.1651459, 146.3995514, -40.6413651, 155.8389130, -194.0040588, 187.0409241
1: -101.3597336, 336.4841309, -107.8246231, 358.2439575, -459.6036987, 444.3087158
2: -147.8714905, 295.7309265, -157.3483429, 314.9838257, -462.8553162, 453.0792847
3: -86.7037125, 355.2321472, -92.2638092, 377.8656616, -464.5693665, 447.4959717
4: -136.0756989, 256.3522644, -144.8023376, 272.9381104, -409.0137939, 401.1546021

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7244032, upper bound: 187.7248576
time: 0.67 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7244032, upper bound: 187.7248810
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -38.3919754, 147.3352661, -40.1432648, 154.0724945, -192.4644623, 187.4785309
1: -101.9445343, 338.7888794, -106.4936600, 354.3966675, -456.3411865, 445.2825317
2: -148.4764557, 297.7420349, -154.7079773, 311.6340637, -460.1105347, 452.4500122
3: -87.1795578, 357.5561218, -91.0647278, 373.6223450, -460.8019104, 448.6208496
4: -136.7099152, 258.0761414, -142.5674591, 269.9847107, -406.6945496, 400.6436157

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7240156, upper bound: 187.7242997
time: 1.10 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250940, upper bound: 187.7254110
time: 0.66 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250940, upper bound: 187.7257867
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.1194954, 150.0627289, -41.6719589, 160.2366028, -199.3560791, 191.7346802
1: -103.6343079, 344.9039001, -110.7073975, 369.5130920, -473.1473999, 455.6112976
2: -150.6259155, 304.1465149, -160.1823425, 323.4183655, -474.0441589, 464.3288574
3: -88.6433640, 363.5794373, -94.6187897, 389.8112183, -478.4545593, 458.1982422
4: -138.7919006, 263.4956360, -147.6698608, 280.3576660, -419.1495361, 411.1654358

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.2973289, 154.8868103, -41.6719589, 160.2366028, -200.5339355, 196.5587463
1: -107.0050964, 356.7510071, -110.7073975, 369.5130920, -476.5181885, 467.4584045
2: -154.9298248, 312.9186707, -160.1823425, 323.4183655, -478.3481445, 473.1010132
3: -91.4977570, 376.5781555, -94.6187897, 389.8112183, -481.3089600, 471.1969604
4: -142.8547821, 271.3110352, -147.6698608, 280.3576660, -423.2124634, 418.9808960

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -39.9541550, 153.3188171, -40.6413651, 155.8389130, -195.7930603, 193.9601746
1: -106.3530579, 352.0562439, -107.8246231, 358.2439575, -464.5970154, 459.8807983
2: -155.0137329, 309.1469727, -157.3483429, 314.9838257, -469.9975586, 466.4953003
3: -90.8584366, 371.8896179, -92.2638092, 377.8656616, -468.7240906, 464.1534424
4: -142.5211029, 267.8211975, -144.8023376, 272.9381104, -415.4592285, 412.6235352

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7242286, upper bound: 187.7244841
time: 0.71 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7242286, upper bound: 187.7249413
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -40.2844429, 154.7667999, -40.1432648, 154.0724945, -194.3569031, 194.9100647
1: -107.2689209, 355.4936829, -106.4936600, 354.3966675, -461.6655579, 461.9873352
2: -156.0757141, 311.8529053, -154.7079773, 311.6340637, -467.7097778, 466.5608521
3: -91.6086960, 375.4405823, -91.0647278, 373.6223450, -465.2310486, 466.5052795
4: -143.5749664, 270.1660767, -142.5674591, 269.9847107, -413.5596313, 412.7335205

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252846, upper bound: 187.7258594
time: 0.69 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7248621, upper bound: 187.7249010
time: 0.70 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7248621, upper bound: 187.7258621
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.3345795, 158.8168793, -41.3863297, 159.1393433, -200.4738922, 200.2032013
1: -109.9803925, 365.3424988, -109.9184875, 367.0632629, -477.0435791, 475.2609253
2: -159.5480957, 319.9708557, -158.9975281, 321.2907715, -480.8388672, 478.9683228
3: -93.9631958, 385.9449158, -93.9397049, 387.1796265, -481.1428223, 479.8845520
4: -146.8587494, 277.3118896, -146.5813446, 278.5128479, -425.3715515, 423.8932495

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252016, upper bound: 187.7258099
time: 0.78 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.7159767, 160.3390350, -41.4825439, 159.5065765, -201.2225494, 201.8215790
1: -110.9884872, 368.7874451, -110.2031555, 367.8280334, -478.8164368, 478.9906006
2: -160.8120270, 323.2605591, -159.4795685, 321.9520264, -482.7640381, 482.7400818
3: -94.7856369, 389.4459534, -94.1892090, 388.0296936, -482.8153076, 483.6351013
4: -148.0745544, 280.1250610, -147.0132599, 279.0867615, -427.1612854, 427.1382751

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253122, upper bound: 187.7259817
time: 0.69 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253098, upper bound: 187.7259990
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -38.1651459, 146.3995514, -43.0866356, 165.0661011, -203.2312469, 189.4861908
1: -101.3597336, 336.4841309, -114.5342636, 379.6184692, -480.9782104, 451.0184021
2: -147.8714905, 295.7309265, -166.3584747, 333.6775818, -481.5490417, 462.0894165
3: -86.7037125, 355.2321472, -97.8645172, 400.7399597, -487.4436646, 453.0966797
4: -136.0756989, 256.3522644, -153.1287537, 289.0673218, -425.1430054, 409.4810181

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7235677, upper bound: 187.7237787
time: 0.69 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249389, upper bound: 187.7253260
time: 0.74 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249389, upper bound: 187.7255916
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -38.3919754, 147.3352661, -42.5347824, 163.4242554, -201.8162231, 189.8700562
1: -101.9445343, 338.7888794, -113.1625366, 376.1094055, -478.0538940, 451.9514160
2: -148.4764557, 297.7420349, -163.6717377, 329.7689514, -478.2453918, 461.4137573
3: -87.1795578, 357.5561218, -96.5832672, 396.7938232, -483.9733582, 454.1394043
4: -136.7099152, 258.0761414, -150.8339844, 285.5761719, -422.2859802, 408.9101257

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7240156, upper bound: 187.7242886
time: 0.80 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251749, upper bound: 187.7254784
time: 0.82 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251749, upper bound: 187.7258259
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -40.4360085, 155.4136658, -43.3119278, 166.6232300, -207.0592194, 198.7255859
1: -107.3941574, 357.9484863, -115.3943024, 383.7673645, -491.1614380, 473.3427429
2: -155.5925446, 313.8864136, -166.8541565, 335.4154053, -491.0079346, 480.7405701
3: -91.8382950, 377.8761597, -98.4820557, 405.3342896, -497.1725769, 476.3582153
4: -143.4478149, 272.1667786, -153.7053223, 290.6419067, -434.0897217, 425.8720703

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268563, upper bound: 187.7271331
time: 0.88 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261352, upper bound: 187.7265084
time: 1.53 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268705, upper bound: 187.7271292
time: 0.90 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268705, upper bound: 187.7271756
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -40.4240646, 155.3548737, -43.4386139, 167.1141357, -207.5381927, 198.7934570
1: -107.3699112, 357.7799683, -115.6725616, 384.8458862, -492.2157288, 473.4525146
2: -155.5592194, 313.7737732, -167.0774536, 336.6412048, -492.2004395, 480.8511658
3: -91.8127289, 377.7077637, -98.7140579, 406.2988281, -498.1115723, 476.4217834
4: -143.4161987, 272.0627441, -153.9809875, 291.6617737, -435.0779419, 426.0437317

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268207, upper bound: 187.7271081
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265561, upper bound: 187.7269578
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -40.0793228, 153.9647217, -44.2105675, 169.9147186, -209.9940491, 198.1752930
1: -106.7526779, 353.4954529, -117.5364227, 390.4844666, -497.2371216, 471.0318604
2: -155.3720703, 310.2699280, -169.1988678, 343.4964294, -498.8684998, 479.4687805
3: -91.1536179, 373.2783203, -100.2381821, 411.7476807, -502.9013062, 473.5164795
4: -142.9090424, 268.7647095, -156.1489105, 297.2363281, -440.1453552, 424.9136353

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7238709, upper bound: 187.7241633
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7240977, upper bound: 187.7243843
time: 0.70 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -40.5245781, 155.6844635, -42.9073334, 164.7347565, -205.2593231, 198.5917816
1: -107.9039001, 357.6199036, -114.0801163, 379.0988770, -487.0027771, 471.6999512
2: -156.9335175, 313.7859192, -164.8737183, 332.9230347, -489.8565674, 478.6596375
3: -92.1424179, 377.6448059, -97.3747253, 399.8435669, -491.9859924, 475.0194702
4: -144.3761749, 271.8290100, -151.9821777, 288.3091736, -432.6853027, 423.8111877

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251101, upper bound: 187.7256736
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252118, upper bound: 187.7259036
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -41.4706764, 159.4295959, -42.7082138, 164.2743225, -205.7449951, 202.1378021
1: -110.2786636, 366.8838196, -113.7740631, 378.6963806, -488.9750366, 480.6578674
2: -159.4913330, 321.5641785, -164.0712280, 330.9793091, -490.4706421, 485.6354065
3: -94.1978226, 387.3689880, -97.0933380, 399.8630066, -494.0608215, 484.4623413
4: -146.9852600, 278.6607361, -151.3323212, 286.7539062, -433.7391357, 429.9930420

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269191, upper bound: 187.7269621
time: 0.69 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
time: 0.78 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274282, upper bound: 187.7274282
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -42.0101967, 161.4943085, -43.3463020, 166.7860565, -208.7962494, 204.8406067
1: -111.7771835, 371.4429321, -115.4366989, 384.0530090, -495.8302002, 486.8796082
2: -161.9033051, 325.5210876, -166.7370605, 335.9248962, -497.8281860, 492.2581482
3: -95.4686584, 392.3119507, -98.5127182, 405.5193176, -500.9879761, 490.8246765
4: -149.1021576, 282.0998230, -153.6667175, 291.0546265, -440.1567993, 435.7665405

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264207, upper bound: 187.7268674
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269792, upper bound: 187.7270019
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -42.2790756, 161.9431763, -39.9390717, 153.3025208, -195.5816040, 201.8822479
1: -112.4335556, 371.8548279, -105.9538116, 352.7402344, -465.1737976, 477.8086243
2: -164.4772491, 326.8182068, -154.0614929, 309.9527893, -474.4300232, 480.8796997
3: -96.1923523, 392.7835999, -90.6229553, 371.9115295, -468.1038818, 483.4065552
4: -151.2476044, 283.1088257, -141.9365845, 268.5572510, -419.8048096, 425.0454102

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7248972, upper bound: 187.7247675
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256589, upper bound: 187.7252404
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.7070274, 167.6367798, -40.7278328, 156.5975952, -200.3046265, 208.3646088
1: -116.1954041, 385.6466675, -108.1544952, 361.2380981, -477.4335022, 493.8011475
2: -169.0041046, 338.4664307, -156.5989227, 316.0985107, -485.1026001, 495.0653687
3: -99.3641281, 407.2857056, -92.4599991, 381.0607910, -480.4248962, 499.7456970
4: -155.6069794, 293.2583008, -144.3376923, 274.0275574, -429.6345215, 437.5960083

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259992, upper bound: 187.7253971
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259992, upper bound: 187.7254180
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -44.9676971, 172.2751312, -39.9390717, 153.3025208, -198.2702179, 212.2142029
1: -119.8964539, 395.6935730, -105.9538116, 352.7402344, -472.6366272, 501.6473694
2: -174.7016144, 347.3181763, -154.0614929, 309.9527893, -484.6544189, 501.3796692
3: -102.4031601, 418.1099548, -90.6229553, 371.9115295, -474.3146973, 508.7329102
4: -160.6975708, 300.7379150, -141.9365845, 268.5572510, -429.2547913, 442.6744995

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249630, upper bound: 187.7250141
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259050, upper bound: 187.7259899
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.4195251, 177.9256134, -40.7278328, 156.5975952, -203.0171051, 218.6534424
1: -123.6620026, 409.0935059, -108.1544952, 361.2380981, -484.8999939, 517.2479248
2: -179.4323425, 359.0392761, -156.5989227, 316.0985107, -495.5308533, 515.6381836
3: -105.6059189, 432.1923828, -92.4599991, 381.0607910, -486.6667175, 524.6523438
4: -165.2159576, 310.9458313, -144.3376923, 274.0275574, -439.2434998, 455.2835083

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_B2_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271292, upper bound: 187.7268705
time: 1.10 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_B2_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271756, upper bound: 187.7268982
time: 1.04 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -42.2790756, 161.9431763, -42.2611084, 162.3383026, -204.6173706, 204.2042847
1: -112.4335556, 371.8548279, -112.4120789, 373.7337341, -486.1672058, 484.2669067
2: -164.4772491, 326.8182068, -162.6969147, 327.4883118, -491.9655762, 489.5151062
3: -96.1923523, 392.7835999, -95.9621506, 394.2851257, -490.4774780, 488.7457581
4: -151.2476044, 283.1088257, -149.8990021, 283.6266785, -434.8742676, 433.0078125

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251486, upper bound: 187.7250596
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251486, upper bound: 187.7250653
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -43.7070274, 167.6367798, -43.0754242, 165.7616577, -209.4686890, 210.7122040
1: -116.1954041, 385.6466675, -114.7028503, 381.8810730, -498.0764771, 500.3495178
2: -169.0041046, 338.4664307, -165.7250519, 333.8023071, -502.8063660, 504.1914673
3: -99.3641281, 407.2857056, -97.8932190, 403.2020569, -502.5661316, 505.1789246
4: -155.6069794, 293.2583008, -152.7065582, 289.2314758, -444.8384399, 445.9648438

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259711, upper bound: 187.7253927
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B1_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261578, upper bound: 187.7254972
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -44.9676971, 172.2751312, -42.2611084, 162.3383026, -207.3059692, 214.5362396
1: -119.8964539, 395.6935730, -112.4120789, 373.7337341, -493.6300964, 508.1056519
2: -174.7016144, 347.3181763, -162.6969147, 327.4883118, -502.1899414, 510.0150757
3: -102.4031601, 418.1099548, -95.9621506, 394.2851257, -496.6882935, 514.0720825
4: -160.6975708, 300.7379150, -149.8990021, 283.6266785, -444.3242493, 450.6369019

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.91 + 417.24 = 420.16 seconds
