## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 3.05840151


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395)
1: (-1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596)
2: (-1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697)
3: (-3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084)
4: (-2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.69 + 1.16 = 1.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982239, upper bound: 3.3982239

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3753023, upper bound: 3.3321438
time: 0.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3944313, upper bound: 3.3944312
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -3.3753023, upper bound: 3.3321438
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -3.3944313, upper bound: 3.3944312

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.2895984, 0.3943615, -1.2466472, 2.2771921, -2.5667903, 1.6410087
1: -0.4578271, 0.5641531, -1.9637374, 3.1950235, -3.6528506, 2.5278900
2: -0.3370600, 0.5814233, -1.3604455, 3.2710245, -3.6080840, 1.9418688
3: -0.6758766, 0.7399411, -3.4595599, 4.0564485, -4.7323251, 4.1995006
4: -0.4528955, 0.7513722, -2.1785955, 4.2066536, -4.6595483, 2.9299674

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.37 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3321438
time: 0.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.0333478, 1.8749698, -1.0739391, 1.9228451, -2.9561930, 2.9489088
1: -1.6493901, 2.6237774, -1.6996270, 2.7132142, -4.3626037, 4.3234034
2: -1.1160958, 2.7230368, -1.1644002, 2.7822537, -3.8983490, 3.8874369
3: -2.9354391, 3.3341267, -2.9835598, 3.4628844, -6.3983235, 6.3176866
4: -1.7549448, 3.5216486, -1.8600720, 3.5994568, -5.3544002, 5.3817201

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3321438, upper bound: 3.3753023
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3321438, upper bound: 3.3944313
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3321438
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -3.3321438, upper bound: 3.3753023
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -3.3321438, upper bound: 3.3944313

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.2895984, 0.3943615, -0.2895984, 0.3943615, -0.6839598, 0.6839598
1: -0.4578271, 0.5641531, -0.4578271, 0.5641531, -1.0219798, 1.0219798
2: -0.3370600, 0.5814233, -0.3370600, 0.5814233, -0.9184830, 0.9184831
3: -0.6758766, 0.7399411, -0.6758766, 0.7399411, -1.4158176, 1.4158175
4: -0.4528955, 0.7513722, -0.4528955, 0.7513722, -1.2042676, 1.2042676

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.26 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.2895984, 0.3943615, -1.0333478, 1.8749698, -2.1645675, 1.4277093
1: -0.4578271, 0.5641531, -1.6493901, 2.6237774, -3.0816045, 2.2135432
2: -0.3370600, 0.5814233, -1.1160958, 2.7230368, -3.0600965, 1.6975191
3: -0.6758766, 0.7399411, -2.9354391, 3.3341267, -4.0100031, 3.6753800
4: -0.4528955, 0.7513722, -1.7549448, 3.5216486, -3.9745436, 2.5063167

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3321438
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3321438
time: 0.28 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.0333478, 1.8749698, -0.2895984, 0.3943615, -1.4277093, 2.1645677
1: -1.6493901, 2.6237774, -0.4578271, 0.5641531, -2.2135432, 3.0816045
2: -1.1160958, 2.7230368, -0.3370600, 0.5814233, -1.6975191, 3.0600965
3: -2.9354391, 3.3341267, -0.6758766, 0.7399411, -3.6753798, 4.0100031
4: -1.7549448, 3.5216486, -0.4528955, 0.7513722, -2.5063167, 3.9745436

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3183662, upper bound: 3.3625951
time: 0.28 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
time: 0.36 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.0333478, 1.8749698, -1.0333478, 1.8749698, -2.9083173, 2.9083176
1: -1.6493901, 2.6237774, -1.6493901, 2.6237774, -4.2731667, 4.2731667
2: -1.1160958, 2.7230368, -1.1160958, 2.7230368, -3.8391325, 3.8391325
3: -2.9354391, 3.3341267, -2.9354391, 3.3341267, -6.2695656, 6.2695656
4: -1.7549448, 3.5216486, -1.7549448, 3.5216486, -5.2765932, 5.2765932

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3183663, upper bound: 3.3652354
time: 0.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3205784
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3321438
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3321438
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3183662, upper bound: 3.3625951
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3183663, upper bound: 3.3652354
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3205784

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1912085, 0.2109509, -0.2281166, 0.2985238, -0.4897323, 0.4390675
1: -0.2706243, 0.2982620, -0.3520592, 0.4246108, -0.6952351, 0.6503212
2: -0.2345207, 0.3197280, -0.2765411, 0.4409244, -0.6754452, 0.5962691
3: -0.3691734, 0.3672814, -0.4958154, 0.5477272, -0.9169006, 0.8630968
4: -0.2326370, 0.4024982, -0.3381903, 0.5589418, -0.7915788, 0.7406885

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4519231, 0.6203508, -0.2312826, 0.3023012, -0.7542241, 0.8516334
1: -0.7154577, 0.8934209, -0.3596209, 0.4322774, -1.1477350, 1.2530417
2: -0.4850966, 0.9528847, -0.2789872, 0.4431193, -0.9282159, 1.2318718
3: -1.1693269, 1.1907833, -0.5034688, 0.5565653, -1.7258922, 1.6942520
4: -0.7152326, 1.2680461, -0.3409680, 0.5605277, -1.2757602, 1.6090140

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1912085, 0.2109509, -0.8948215, 1.5723711, -1.7635796, 1.1057724
1: -0.2706243, 0.2982620, -1.4314557, 2.2096181, -2.4802423, 1.7297175
2: -0.2345207, 0.3197280, -0.9551914, 2.3049178, -2.5394385, 1.2749193
3: -0.3691734, 0.3672814, -2.5370455, 2.8204222, -3.1895957, 2.9043269
4: -0.2326370, 0.4024982, -1.4911149, 2.9982629, -3.2308998, 1.8936130

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2829782, upper bound: 3.1939054
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3135975
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3692532, upper bound: 3.3251391
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4519231, 0.6203508, -0.8392967, 1.4580309, -1.9099538, 1.4596474
1: -0.7154577, 0.8934209, -1.3524570, 2.0678201, -2.7832773, 2.2458777
2: -0.4850966, 0.9528847, -0.9002258, 2.1351390, -2.6202352, 1.8531102
3: -1.1693269, 1.1907833, -2.3765635, 2.6455564, -3.8148832, 3.5673468
4: -0.7152326, 1.2680461, -1.4056300, 2.7856278, -3.5008602, 2.6736755

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3625952, upper bound: 3.3183663
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3753012, 0.5288189, -0.2507620, 0.3365805, -0.7118816, 0.7795810
1: -0.6084703, 0.7558654, -0.3941799, 0.4819908, -1.0904610, 1.1500453
2: -0.4198778, 0.7994800, -0.2989590, 0.4936025, -0.9134802, 1.0984390
3: -0.9523419, 0.9947283, -0.5622559, 0.6275445, -1.5798861, 1.5569842
4: -0.5820951, 1.0479591, -0.3844816, 0.6305283, -1.2126232, 1.4324408

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.27 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3183662, upper bound: 3.3625951
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.2309401, 0.3020085, -0.9208838, 1.1963177
1: -1.0178568, 1.3381594, -0.3573236, 0.4308809, -1.4487376, 1.6954831
2: -0.6695868, 1.4706283, -0.2781435, 0.4461296, -1.1157162, 1.7487717
3: -1.7465825, 1.7551888, -0.5060763, 0.5565612, -2.3031437, 2.2612653
4: -1.0122970, 1.9336305, -0.3456290, 0.5663463, -1.5786433, 2.2792594

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3753012, 0.5288189, -0.9496339, 1.6915393, -2.0668399, 1.4784528
1: -0.6084703, 0.7558654, -1.5198787, 2.3741374, -2.9826076, 2.2757442
2: -0.4198778, 0.7994800, -1.0203989, 2.4693260, -2.8892038, 1.8198786
3: -0.9523419, 0.9947283, -2.6952519, 3.0279951, -3.9803369, 3.6899791
4: -0.5820951, 1.0479591, -1.5999130, 3.2068162, -3.7889113, 2.6478715

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.8905395, 1.5623000, -2.1811748, 1.8559169
1: -1.0178568, 1.3381594, -1.4336659, 2.1884227, -3.2062795, 2.7718251
2: -0.6695868, 1.4706283, -0.9582002, 2.2914517, -2.9610384, 2.4288285
3: -1.7465825, 1.7551888, -2.5283735, 2.8014264, -4.5480089, 4.2835617
4: -1.0122970, 1.9336305, -1.4965149, 2.9785175, -3.9908147, 3.4301453

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.32 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3135975
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3692532, upper bound: 3.3251391
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3625952, upper bound: 3.3183663
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3183662, upper bound: 3.3625951
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3122129
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.32
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1912085, 0.2109509, -0.1912085, 0.2109509, -0.4021594, 0.4021594
1: -0.2706243, 0.2982620, -0.2706243, 0.2982620, -0.5688863, 0.5688863
2: -0.2345207, 0.3197280, -0.2345207, 0.3197280, -0.5542487, 0.5542487
3: -0.3691734, 0.3672814, -0.3691734, 0.3672814, -0.7364548, 0.7364548
4: -0.2326370, 0.4024982, -0.2326370, 0.4024982, -0.6351352, 0.6351352

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2639925
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1912085, 0.2109509, -0.3982752, 0.5545195, -0.7457280, 0.6092261
1: -0.2706243, 0.2982620, -0.6593187, 0.8071097, -1.0777340, 0.9575807
2: -0.2345207, 0.3197280, -0.4551491, 0.7819526, -1.0164733, 0.7748770
3: -0.3691734, 0.3672814, -0.9516394, 1.0879471, -1.4571205, 1.3189209
4: -0.2326370, 0.4024982, -0.6564777, 1.0258938, -1.2585309, 1.0589759

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2938263
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4519231, 0.6203508, -0.1912085, 0.2109509, -0.6628739, 0.8115593
1: -0.7154577, 0.8934209, -0.2706243, 0.2982620, -1.0137197, 1.1640452
2: -0.4850966, 0.9528847, -0.2345207, 0.3197280, -0.8048245, 1.1874055
3: -1.1693269, 1.1907833, -0.3691734, 0.3672814, -1.5366082, 1.5599567
4: -0.7152326, 1.2680461, -0.2326370, 0.4024982, -1.1177306, 1.5006832

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054651
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4519231, 0.6203508, -0.4002510, 0.5375090, -0.9894320, 1.0206017
1: -0.7154577, 0.8934209, -0.6595496, 0.8165386, -1.5319963, 1.5529705
2: -0.4850966, 0.9528847, -0.4584243, 0.7346345, -1.2197310, 1.4113090
3: -1.1693269, 1.1907833, -0.8917058, 1.0986850, -2.2680120, 2.0824890
4: -0.7152326, 1.2680461, -0.6609793, 0.9406634, -1.6558959, 1.9290254

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054680
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1912085, 0.2109509, -0.5354136, 0.7824069, -0.9736154, 0.7463645
1: -0.2706243, 0.2982620, -0.8653058, 1.0692652, -1.3398895, 1.1635677
2: -0.2345207, 0.3197280, -0.5768373, 1.2165308, -1.4510516, 0.8965650
3: -0.3691734, 0.3672814, -1.4866035, 1.4143922, -1.7835656, 1.8538849
4: -0.2326370, 0.4024982, -0.8476683, 1.6256173, -1.8582543, 1.2501663

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3112246
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3135974
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1694227, 0.1482669, -0.5740923, 0.8627723, -1.0321951, 0.7223592
1: -0.2317611, 0.2145776, -0.9296207, 1.2116103, -1.4433714, 1.1441983
2: -0.2058225, 0.2303603, -0.6147872, 1.3206720, -1.5264946, 0.8451474
3: -0.3057782, 0.2565590, -1.5974710, 1.5889094, -1.8946875, 1.8540300
4: -0.1899571, 0.2856969, -0.9161484, 1.7554853, -1.9454424, 1.2018452

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3440960, upper bound: 3.3015846
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4159712, 0.5720838, -0.3253944, 0.4562526, -0.8722238, 0.8974780
1: -0.6589215, 0.8236436, -0.5269880, 0.6540714, -1.3129928, 1.3506316
2: -0.4504858, 0.8728558, -0.3711399, 0.6824883, -1.1329741, 1.2439955
3: -1.0644279, 1.0946593, -0.7994739, 0.8534504, -1.9178782, 1.8941330
4: -0.6557244, 1.1559072, -0.4960128, 0.8813407, -1.5370648, 1.6519200

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3534641, upper bound: 3.3095118
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3750907, 0.5151983, -0.4756438, 0.6661956, -1.0412862, 0.9908422
1: -0.6011347, 0.7432693, -0.7971063, 0.9370974, -1.5382318, 1.5403756
2: -0.4144570, 0.7740390, -0.5279092, 1.0286267, -1.4430836, 1.3019481
3: -0.9491341, 0.9867679, -1.3019695, 1.2563683, -2.2055025, 2.2887373
4: -0.5930573, 1.0221404, -0.7658277, 1.3805962, -1.9736530, 1.7879682

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3048863, upper bound: 3.2880725
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2897610, 0.4061280, -0.1805916, 0.1821264, -0.4718874, 0.5867196
1: -0.4672413, 0.5800911, -0.2518537, 0.2591882, -0.7264295, 0.8319448
2: -0.3363055, 0.6044844, -0.2208168, 0.2780043, -0.6143097, 0.8253012
3: -0.6918739, 0.7532626, -0.3350123, 0.3136986, -1.0055726, 1.0882750
4: -0.4368910, 0.7733765, -0.2091804, 0.3479619, -0.7848529, 0.9825569

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.28 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.27 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3253944, 0.4562526, -0.3727903, 0.5206936, -0.8460880, 0.8290429
1: -0.5269880, 0.6540714, -0.6165087, 0.7595952, -1.2865829, 1.2705801
2: -0.3711399, 0.6824883, -0.4286546, 0.7308654, -1.1020050, 1.1111430
3: -0.7994739, 0.8534504, -0.8819786, 1.0189972, -1.8184711, 1.7354290
4: -0.4960128, 0.8813407, -0.6121606, 0.9518200, -1.4478328, 1.4935013

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2697185, upper bound: 3.3461173
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2697185, upper bound: 3.3625951
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.1562187, 0.0892744, -0.7081497, 1.1215963
1: -1.0178568, 1.3381594, -0.2067900, 0.1396068, -1.1574637, 1.5449494
2: -0.6695868, 1.4706283, -0.1889970, 0.1377370, -0.8073239, 1.6596253
3: -1.7465825, 1.7551888, -0.2654234, 0.1652304, -1.9118128, 2.0206122
4: -1.0122970, 1.9336305, -0.1719097, 0.1611379, -1.1734350, 2.1055403

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2939146, upper bound: 3.3122129
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3121866
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.1972768, 0.2288733, -0.8477486, 1.1626544
1: -1.0178568, 1.3381594, -0.2833577, 0.3218512, -1.3397081, 1.6215172
2: -0.6695868, 1.4706283, -0.2425594, 0.3484467, -1.0180335, 1.7131877
3: -1.7465825, 1.7551888, -0.3993823, 0.4013456, -2.1479280, 2.1545711
4: -1.0122970, 1.9336305, -0.2542248, 0.4395830, -1.4518801, 2.1878552

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2939146, upper bound: 3.3122129
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3121866
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3753012, 0.5288189, -0.3753012, 0.5288189, -0.9041201, 0.9041200
1: -0.6084703, 0.7558654, -0.6084703, 0.7558654, -1.3643357, 1.3643357
2: -0.4198778, 0.7994800, -0.4198778, 0.7994800, -1.2193577, 1.2193577
3: -0.9523419, 0.9947283, -0.9523419, 0.9947283, -1.9470696, 1.9470699
4: -0.5820951, 1.0479591, -0.5820951, 1.0479591, -1.6300540, 1.6300540

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3293443, upper bound: 3.3652353
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3232136, upper bound: 3.3366522
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3753012, 0.5288189, -0.6188753, 0.9653776, -1.3406787, 1.1476941
1: -0.6084703, 0.7558654, -1.0178568, 1.3381594, -1.9466296, 1.7737222
2: -0.4198778, 0.7994800, -0.6695868, 1.4706283, -1.8905060, 1.4690667
3: -0.9523419, 0.9947283, -1.7465825, 1.7551888, -2.7075305, 2.7413108
4: -0.5820951, 1.0479591, -1.0122970, 1.9336305, -2.5157256, 2.0602558

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3293443, upper bound: 3.3652353
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3232136, upper bound: 3.3366522
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.3753012, 0.5288189, -1.1476941, 1.3406787
1: -1.0178568, 1.3381594, -0.6084703, 0.7558654, -1.7737221, 1.9466296
2: -0.6695868, 1.4706283, -0.4198778, 0.7994800, -1.4690667, 1.8905060
3: -1.7465825, 1.7551888, -0.9523419, 0.9947283, -2.7413106, 2.7075305
4: -1.0122970, 1.9336305, -0.5820951, 1.0479591, -2.0602555, 2.5157256

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3133814
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.27 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6188753, 0.9653776, -0.6188753, 0.9653776, -1.5842528, 1.5842528
1: -1.0178568, 1.3381594, -1.0178568, 1.3381594, -2.3560159, 2.3560159
2: -0.6695868, 1.4706283, -0.6695868, 1.4706283, -2.1402152, 2.1402152
3: -1.7465825, 1.7551888, -1.7465825, 1.7551888, -3.5017715, 3.5017715
4: -1.0122970, 1.9336305, -1.0122970, 1.9336305, -2.9459276, 2.9459276

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3133814
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.79 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2639925
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2938263
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054651
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054680
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3112246
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3304820, upper bound: 3.3135974
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3440960, upper bound: 3.3015846
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3122129, upper bound: 3.3049880
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.2697185, upper bound: 3.3461173
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.2697185, upper bound: 3.3625951
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.2939146, upper bound: 3.3122129
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3121866
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.2939146, upper bound: 3.3122129
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3049880, upper bound: 3.3121866
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3293443, upper bound: 3.3652353
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3232136, upper bound: 3.3366522
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3293443, upper bound: 3.3652353
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3232136, upper bound: 3.3366522
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3133814
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3133814
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.79
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0676768, 0.0203616, -0.1805916, 0.1821264, -0.2498033, 0.2009531
1: -0.0773866, 0.0391565, -0.2518537, 0.2591882, -0.3365748, 0.2910102
2: -0.0644345, 0.0282484, -0.2208168, 0.2780043, -0.3424388, 0.2490652
3: -0.0712000, 0.0471544, -0.3350123, 0.3136986, -0.3848985, 0.3821667
4: -0.0532835, 0.0354563, -0.2091804, 0.3479619, -0.4012455, 0.2446367

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0610602, upper bound: 3.1366150
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2639925
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.1755409, 0.1610399, -0.3260270, 0.2930316
1: -0.2215959, 0.1765458, -0.2416268, 0.2308512, -0.4524470, 0.4181726
2: -0.1996021, 0.1877475, -0.2137747, 0.2493992, -0.4490014, 0.4015222
3: -0.2914082, 0.2083298, -0.3209178, 0.2744416, -0.5658498, 0.5292476
4: -0.1816220, 0.2294925, -0.1968306, 0.3111755, -0.4927976, 0.4263231

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0676768, 0.0203616, -0.3697518, 0.5146146, -0.5822914, 0.3901133
1: -0.0773866, 0.0391565, -0.6097153, 0.7480660, -0.8254526, 0.6488719
2: -0.0644345, 0.0282484, -0.4246119, 0.7245451, -0.7889796, 0.4528603
3: -0.0712000, 0.0471544, -0.8724087, 1.0049496, -1.0761496, 0.9195631
4: -0.0532835, 0.0354563, -0.6050246, 0.9449091, -0.9981926, 0.6404809

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2363997, upper bound: 3.2869785
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.3435573, 0.4733745, -0.6383616, 0.4610480
1: -0.2215959, 0.1765458, -0.5617366, 0.6859095, -0.9075054, 0.7382824
2: -0.1996021, 0.1877475, -0.3950204, 0.6662990, -0.8659012, 0.5827679
3: -0.2914082, 0.2083298, -0.7999631, 0.9179860, -1.2093942, 1.0082929
4: -0.1816220, 0.2294925, -0.5554047, 0.8641331, -1.0457551, 0.7848971

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.1912085, 0.2109509, -0.4668896, 0.5327840
1: -0.3946697, 0.4863652, -0.2706243, 0.2982620, -0.6929317, 0.7569895
2: -0.2981569, 0.5062003, -0.2345207, 0.3197280, -0.6178848, 0.7407210
3: -0.5855743, 0.6284050, -0.3691734, 0.3672814, -0.9528557, 0.9975784
4: -0.3760676, 0.6415557, -0.2326370, 0.4024982, -0.7785658, 0.8741927

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2869785, upper bound: 3.2363997
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041510, upper bound: 3.3042233
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041510, upper bound: 3.3054651
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.1694227, 0.1482669, -0.5569730, 0.7341877
1: -0.6539204, 0.7966581, -0.2317611, 0.2145776, -0.8684979, 1.0284193
2: -0.4457314, 0.8629469, -0.2058225, 0.2303603, -0.6760917, 1.0687695
3: -1.0570911, 1.0608249, -0.3057782, 0.2565590, -1.3136501, 1.3666030
4: -0.6360155, 1.1408229, -0.1899571, 0.2856969, -0.9217123, 1.3307800

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052334, upper bound: 3.3086207
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052334, upper bound: 3.3098627
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.4002510, 0.5375090, -0.7934476, 0.7418266
1: -0.3946697, 0.4863652, -0.6595496, 0.8165386, -1.2112083, 1.1459148
2: -0.2981569, 0.5062003, -0.4584243, 0.7346345, -1.0327914, 0.9646246
3: -0.5855743, 0.6284050, -0.8917058, 1.0986850, -1.6842594, 1.5201108
4: -0.3760676, 0.6415557, -0.6609793, 0.9406634, -1.3167310, 1.3025349

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043864, upper bound: 3.3043662
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043864, upper bound: 3.3054680
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.3373367, 0.4591842, -0.8678904, 0.9021017
1: -0.6539204, 0.7966581, -0.5522138, 0.6849697, -1.3388901, 1.3488719
2: -0.4457314, 0.8629469, -0.3905955, 0.6308199, -1.0765512, 1.2535423
3: -1.0570911, 1.0608249, -0.7496297, 0.9152554, -1.9723465, 1.8104546
4: -0.6360155, 1.1408229, -0.5483062, 0.8039945, -1.4400101, 1.6891291

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1506878, 0.0907005, -0.5354136, 0.7824069, -0.9330947, 0.6261141
1: -0.2101478, 0.1401098, -0.8653058, 1.0692652, -1.2794131, 1.0054153
2: -0.1937296, 0.1382380, -0.5768373, 1.2165308, -1.4102604, 0.7150754
3: -0.2145611, 0.1737585, -1.4866035, 1.4143922, -1.6289533, 1.6603620
4: -0.1262522, 0.1747571, -0.8476683, 1.6256173, -1.7518694, 1.0224252

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3093662
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3112246
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1706024, 0.1344135, -0.5354136, 0.7824069, -0.9530094, 0.6698271
1: -0.2284564, 0.1925970, -0.8653058, 1.0692652, -1.2977216, 1.0579026
2: -0.2065775, 0.1987212, -0.5768373, 1.2165308, -1.4231082, 0.7755585
3: -0.2968557, 0.2297280, -1.4866035, 1.4143922, -1.7112479, 1.7163315
4: -0.1908983, 0.2396057, -0.8476683, 1.6256173, -1.8165156, 1.0872737

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3116771
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3135974
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1442973, 0.0641716, -0.5740923, 0.8627723, -1.0070697, 0.6382639
1: -0.1991781, 0.1041099, -0.9296207, 1.2116103, -1.4107884, 1.0337306
2: -0.1851572, 0.0952137, -0.6147872, 1.3206720, -1.5058292, 0.7100008
3: -0.1959259, 0.1303892, -1.5974710, 1.5889094, -1.7848353, 1.7278602
4: -0.1161131, 0.1196624, -0.9161484, 1.7554853, -1.8715985, 1.0358107

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3440959, upper bound: 3.3015846
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3440959, upper bound: 3.3015846
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1469732, 0.0585246, -0.4673221, 0.6641812, -0.8111544, 0.5258466
1: -0.2018635, 0.0926680, -0.7591437, 0.9236031, -1.1254666, 0.8518116
2: -0.1884660, 0.0827792, -0.5063106, 1.0216051, -1.2100711, 0.5890898
3: -0.1947707, 0.1182943, -1.2630637, 1.2244874, -1.4192581, 1.3813579
4: -0.1158141, 0.1031256, -0.7266700, 1.3647677, -1.4805818, 0.8297956

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1850611, 0.2096545, -0.3253944, 0.4562526, -0.6413137, 0.5350488
1: -0.2660490, 0.2913542, -0.5269880, 0.6540714, -0.9201204, 0.8183422
2: -0.2290718, 0.3008571, -0.3711399, 0.6824883, -0.9115601, 0.6719970
3: -0.3428191, 0.3601812, -0.7994739, 0.8534504, -1.1962695, 1.1596551
4: -0.2267372, 0.3690930, -0.4960128, 0.8813407, -1.1080779, 0.8651057

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3461173, upper bound: 3.2697185
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3461173, upper bound: 3.3183662
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3088257, 0.4204044, -0.3253944, 0.4562526, -0.7650784, 0.7457986
1: -0.4969721, 0.6138726, -0.5269880, 0.6540714, -1.1510434, 1.1408607
2: -0.3497591, 0.6176917, -0.3711399, 0.6824883, -1.0322474, 0.9888313
3: -0.7490399, 0.8086340, -0.7994739, 0.8534504, -1.6024903, 1.6081077
4: -0.4839897, 0.8007323, -0.4960128, 0.8813407, -1.3653302, 1.2967451

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3461173, upper bound: 3.2697185
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3461173, upper bound: 3.3183662
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1850611, 0.2096545, -0.4756438, 0.6661956, -0.8512567, 0.6852983
1: -0.2660490, 0.2913542, -0.7971063, 0.9370974, -1.2031465, 1.0884604
2: -0.2290718, 0.3008571, -0.5279092, 1.0286267, -1.2576985, 0.8287662
3: -0.3428191, 0.3601812, -1.3019695, 1.2563683, -1.5991874, 1.6621506
4: -0.2267372, 0.3690930, -0.7658277, 1.3805962, -1.6073334, 1.1349207

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.2939146
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121866, upper bound: 3.3049880
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3088257, 0.4204044, -0.4756438, 0.6661956, -0.9750212, 0.8960481
1: -0.4969721, 0.6138726, -0.7971063, 0.9370974, -1.4340694, 1.4109789
2: -0.3497591, 0.6176917, -0.5279092, 1.0286267, -1.3783858, 1.1456007
3: -0.7490399, 0.8086340, -1.3019695, 1.2563683, -2.0054080, 2.1106033
4: -0.4839897, 0.8007323, -0.7658277, 1.3805962, -1.8645856, 1.5665600

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122129, upper bound: 3.2939146
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121866, upper bound: 3.3049880
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.1805916, 0.1821264, -0.4032331, 0.4581523
1: -0.3221700, 0.3887413, -0.2518537, 0.2591882, -0.5813582, 0.6405950
2: -0.2730529, 0.4024681, -0.2208168, 0.2780043, -0.5510572, 0.6232849
3: -0.4342754, 0.4909782, -0.3350123, 0.3136986, -0.7479740, 0.8259904
4: -0.2787128, 0.4985040, -0.2091804, 0.3479619, -0.6266748, 0.7076845

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887059, upper bound: 3.2758720
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.28 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6052738, 0.8398099, -0.1805916, 0.1821264, -0.7874003, 1.0204015
1: -0.9567434, 1.2019296, -0.2518537, 0.2591882, -1.2159315, 1.4537833
2: -0.6341691, 1.3133148, -0.2208168, 0.2780043, -0.9121734, 1.5341315
3: -1.6088847, 1.6106904, -0.3350123, 0.3136986, -1.9225829, 1.9457027
4: -0.9545741, 1.7640181, -0.2091804, 0.3479619, -1.3025358, 1.9731985

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887059, upper bound: 3.2758720
time: 0.27 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.3727903, 0.5206936, -0.7418002, 0.6503510
1: -0.3221700, 0.3887413, -0.6165087, 0.7595952, -1.0817652, 1.0052500
2: -0.2730529, 0.4024681, -0.4286546, 0.7308654, -1.0039184, 0.8311228
3: -0.4342754, 0.4909782, -0.8819786, 1.0189972, -1.4532726, 1.3729568
4: -0.2787128, 0.4985040, -0.6121606, 0.9518200, -1.2305329, 1.1106646

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887059, upper bound: 3.3422717
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3461173
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3461173
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6227801, 0.8648329, -0.3727903, 0.5206936, -1.1434736, 1.2376232
1: -0.9949624, 1.2503651, -0.6165087, 0.7595952, -1.7545574, 1.8668737
2: -0.6539169, 1.3625121, -0.4286546, 0.7308654, -1.3847823, 1.7911668
3: -1.6873388, 1.6740556, -0.8819786, 1.0189972, -2.7063360, 2.5560341
4: -0.9924154, 1.8411534, -0.6121606, 0.9518200, -1.9442354, 2.4533141

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887059, upper bound: 3.3459832
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3557151
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3557151
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7091317, 1.1715200, -0.1424455, 0.0682273, -0.7773590, 1.3139656
1: -1.1533722, 1.6230139, -0.1980544, 0.1071797, -1.2605518, 1.8210683
2: -0.7592056, 1.7569205, -0.1835787, 0.1010061, -0.8602116, 1.9404992
3: -2.0259933, 2.1003478, -0.1961785, 0.1336375, -2.1596303, 2.2965262
4: -1.1636406, 2.2863564, -0.1159588, 0.1256564, -1.2892970, 2.4023154

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4738564, 0.6639866, -0.1562187, 0.0892744, -0.5631308, 0.8202053
1: -0.7879710, 0.9146689, -0.2067900, 0.1396068, -0.9275778, 1.1214589
2: -0.5224863, 1.0312799, -0.1889970, 0.1377370, -0.6602234, 1.2202770
3: -1.3006039, 1.2264051, -0.2654234, 0.1652304, -1.4658343, 1.4918286
4: -0.7528779, 1.3891870, -0.1719097, 0.1611379, -0.9140158, 1.5610967

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.7091317, 1.1715200, -0.1874377, 0.2015967, -0.9107284, 1.3589578
1: -1.1533722, 1.6230139, -0.2656229, 0.2850674, -1.4384396, 1.8886368
2: -0.7592056, 1.7569205, -0.2297924, 0.3094583, -1.0686638, 1.9867128
3: -2.0259933, 2.1003478, -0.3608987, 0.3493154, -2.3753083, 2.4612465
4: -1.1636406, 2.2863564, -0.2272224, 0.3885824, -1.5522230, 2.5135789

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4738564, 0.6639866, -0.1972768, 0.2288733, -0.7027296, 0.8612634
1: -0.7879710, 0.9146689, -0.2833577, 0.3218512, -1.1098222, 1.1980265
2: -0.5224863, 1.0312799, -0.2425594, 0.3484467, -0.8709329, 1.2738392
3: -1.3006039, 1.2264051, -0.3993823, 0.4013456, -1.7019494, 1.6257875
4: -0.7528779, 1.3891870, -0.2542248, 0.4395830, -1.1924607, 1.6434118

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3456418, 0.4810718, -0.3084040, 0.4328659, -0.7785076, 0.7894757
1: -0.5610024, 0.6868975, -0.4976889, 0.6205016, -1.1815040, 1.1845864
2: -0.3901013, 0.7285742, -0.3543311, 0.6460052, -1.0361066, 1.0829053
3: -0.8672440, 0.8995361, -0.7454625, 0.8085792, -1.6758232, 1.6449986
4: -0.5257008, 0.9459569, -0.4695772, 0.8310735, -1.3567743, 1.4155341

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2695478, 0.3760729, -0.3753012, 0.5288189, -0.7983667, 0.7513741
1: -0.4326958, 0.5375808, -0.6084703, 0.7558654, -1.1885612, 1.1460508
2: -0.3200988, 0.5548421, -0.4198778, 0.7994800, -1.1195787, 0.9747199
3: -0.6285393, 0.6923354, -0.9523419, 0.9947283, -1.6232675, 1.6446772
4: -0.3963178, 0.7018194, -0.5820951, 1.0479591, -1.4442769, 1.2839143

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3456418, 0.4810718, -0.5400299, 0.7867532, -1.1323950, 1.0211017
1: -0.5610024, 0.6868975, -0.8933331, 1.0940363, -1.6550387, 1.5802306
2: -0.3901013, 0.7285742, -0.5893232, 1.2207618, -1.6108631, 1.3178973
3: -0.8672440, 0.8995361, -1.5047594, 1.4573568, -2.3246007, 2.4042954
4: -0.5257008, 0.9459569, -0.8745193, 1.6258235, -2.1515243, 1.8204763

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3226378, upper bound: 3.3549587
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3211579, upper bound: 3.3502041
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2695478, 0.3760729, -0.6188753, 0.9653776, -1.2349253, 0.9949480
1: -0.4326958, 0.5375808, -1.0178568, 1.3381594, -1.7708553, 1.5554374
2: -0.3200988, 0.5548421, -0.6695868, 1.4706283, -1.7907270, 1.2244289
3: -0.6285393, 0.6923354, -1.7465825, 1.7551888, -2.3837280, 2.4389179
4: -0.3963178, 0.7018194, -1.0122970, 1.9336305, -2.3299484, 1.7141165

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3151204, upper bound: 3.3209046
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4467238, 0.6215951, -0.3753012, 0.5288189, -0.9755427, 0.9968962
1: -0.7328915, 0.8427354, -0.6084703, 0.7558654, -1.4887569, 1.4512056
2: -0.4924225, 0.9656458, -0.4198778, 0.7994800, -1.2919024, 1.3855237
3: -1.2085794, 1.1289421, -0.9523419, 0.9947283, -2.2033076, 2.0812838
4: -0.7015048, 1.3040695, -0.5820951, 1.0479591, -1.7494637, 1.8861645

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3342106, upper bound: 3.3182551
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208495, upper bound: 3.3151464
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4117214, 0.5673239, -0.2613925, 0.3674621, -0.7791834, 0.8287163
1: -0.6853573, 0.7768623, -0.4228539, 0.5239409, -1.2092981, 1.1997160
2: -0.4597958, 0.8666953, -0.3120625, 0.5395492, -0.9993450, 1.1787578
3: -1.0987918, 1.0422142, -0.6065777, 0.6781660, -1.7769576, 1.6487918
4: -0.6412443, 1.1624334, -0.3911900, 0.6840116, -1.3252559, 1.5536233

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3261272, upper bound: 3.3154022
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4467238, 0.6215951, -0.6188753, 0.9653776, -1.4121013, 1.2404703
1: -0.7328915, 0.8427354, -1.0178568, 1.3381594, -2.0710504, 1.8605920
2: -0.4924225, 0.9656458, -0.6695868, 1.4706283, -1.9630504, 1.6352326
3: -1.2085794, 1.1289421, -1.7465825, 1.7551888, -2.9637682, 2.8755245
4: -0.7015048, 1.3040695, -1.0122970, 1.9336305, -2.6351352, 2.3163664

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4117214, 0.5673239, -0.5122431, 0.7316685, -1.1433899, 1.0795670
1: -0.6853573, 0.7768623, -0.8493786, 1.0150850, -1.7004421, 1.6262407
2: -0.4597958, 0.8666953, -0.5609675, 1.1394755, -1.5992713, 1.4276628
3: -1.0987918, 1.0422142, -1.4193451, 1.3587923, -2.4575841, 2.4615593
4: -0.6412443, 1.1624334, -0.8258383, 1.5242116, -2.1654556, 1.9882717

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132705, upper bound: 3.3125449
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.45 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.0610602, upper bound: 3.1366150
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1864269, upper bound: 3.2639925
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3041510, upper bound: 3.3042233
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3041510, upper bound: 3.3054651
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3052334, upper bound: 3.3086207
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3052334, upper bound: 3.3098627
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3043864, upper bound: 3.3043662
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3043864, upper bound: 3.3054680
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3093662
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3112246
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3116771
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3282178, upper bound: 3.3135974
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3440959, upper bound: 3.3015846
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3440959, upper bound: 3.3015846
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3394338, upper bound: 3.2841771
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3461173, upper bound: 3.2697185
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3461173, upper bound: 3.3183662
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3461173, upper bound: 3.2697185
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3461173, upper bound: 3.3183662
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3122129, upper bound: 3.2939146
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3121866, upper bound: 3.3049880
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3122129, upper bound: 3.2939146
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3121866, upper bound: 3.3049880
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.2829782
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3461173
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3461173
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3557151
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.1939054, upper bound: 3.3557151
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3454026, upper bound: 3.3453012
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3226378, upper bound: 3.3549587
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3211579, upper bound: 3.3502041
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3151204, upper bound: 3.3209046
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3342106, upper bound: 3.3182551
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3208495, upper bound: 3.3151464
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3261272, upper bound: 3.3154022
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3132705, upper bound: 3.3125449
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0652147, 0.0083182, -0.1520702, 0.0857642, -0.1509789, 0.1603884
1: -0.0734224, 0.0195891, -0.1991930, 0.1328123, -0.2062347, 0.2187821
2: -0.0614410, 0.0105155, -0.1822435, 0.1380647, -0.1995058, 0.1927589
3: -0.0668020, 0.0271159, -0.2571985, 0.1591592, -0.2259611, 0.2843143
4: -0.0479105, 0.0142335, -0.1675957, 0.1625305, -0.2104410, 0.1818293

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0676768, 0.0203616, -0.1718930, 0.1522878, -0.2199646, 0.1922546
1: -0.0773866, 0.0391565, -0.2361045, 0.2212102, -0.2985969, 0.2752611
2: -0.0644345, 0.0282484, -0.2093908, 0.2357066, -0.3001412, 0.2376392
3: -0.0712000, 0.0471544, -0.3105830, 0.2643077, -0.3355076, 0.3577374
4: -0.0532835, 0.0354563, -0.1935923, 0.2920630, -0.3453465, 0.2290486

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9085425, upper bound: 2.9832719
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1498572, upper bound: 3.2329620
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1498572, upper bound: 3.2639925
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.0676768, 0.0203616, -0.1853487, 0.1851676
1: -0.2215959, 0.1765458, -0.0773866, 0.0391565, -0.2607524, 0.2539324
2: -0.1996021, 0.1877475, -0.0644345, 0.0282484, -0.2278505, 0.2521820
3: -0.2914082, 0.2083298, -0.0712000, 0.0471544, -0.3385626, 0.2795298
4: -0.1816220, 0.2294925, -0.0532835, 0.0354563, -0.2170783, 0.2827760

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0939294, upper bound: 3.0463998
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.1649871, 0.1174907, -0.2824779, 0.2824779
1: -0.2215959, 0.1765458, -0.2215959, 0.1765458, -0.3981417, 0.3981417
2: -0.1996021, 0.1877475, -0.1996021, 0.1877475, -0.3873496, 0.3873496
3: -0.2914082, 0.2083298, -0.2914082, 0.2083298, -0.4997380, 0.4997380
4: -0.1816220, 0.2294925, -0.1816220, 0.2294925, -0.4111145, 0.4111145

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0939294, upper bound: 3.0463998
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0676768, 0.0203616, -0.1842154, 0.2073062, -0.2749831, 0.2045770
1: -0.0773866, 0.0391565, -0.2644064, 0.2852268, -0.3626135, 0.3035629
2: -0.0644345, 0.0282484, -0.2280039, 0.2978324, -0.3622669, 0.2562523
3: -0.0712000, 0.0471544, -0.3391942, 0.3525377, -0.4237376, 0.3863487
4: -0.0532835, 0.0354563, -0.2241528, 0.3658484, -0.4191319, 0.2596091

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1443273, upper bound: 3.2233479
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2503868, upper bound: 3.2938263
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0676768, 0.0203616, -0.2910421, 0.3949972, -0.4626741, 0.3114037
1: -0.0773866, 0.0391565, -0.4709392, 0.5766153, -0.6540020, 0.5100957
2: -0.0644345, 0.0282484, -0.3379044, 0.5531360, -0.6175705, 0.3661528
3: -0.0712000, 0.0471544, -0.6551671, 0.7631698, -0.8343698, 0.7023215
4: -0.0532835, 0.0354563, -0.4598577, 0.7066551, -0.7599387, 0.4953140

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1443273, upper bound: 3.2233479
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2503868, upper bound: 3.2938263
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.1842154, 0.2073062, -0.3722934, 0.3017061
1: -0.2215959, 0.1765458, -0.2644064, 0.2852268, -0.5068227, 0.4409522
2: -0.1996021, 0.1877475, -0.2280039, 0.2978324, -0.4974345, 0.4157514
3: -0.2914082, 0.2083298, -0.3391942, 0.3525377, -0.6439459, 0.5475241
4: -0.1816220, 0.2294925, -0.2241528, 0.3658484, -0.5474705, 0.4536453

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0936158, upper bound: 3.0463735
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1649871, 0.1174907, -0.2910421, 0.3949972, -0.5599843, 0.4085329
1: -0.2215959, 0.1765458, -0.4709392, 0.5766153, -0.7982112, 0.6474850
2: -0.1996021, 0.1877475, -0.3379044, 0.5531360, -0.7527381, 0.5256519
3: -0.2914082, 0.2083298, -0.6551671, 0.7631698, -1.0545781, 0.8634969
4: -0.1816220, 0.2294925, -0.4598577, 0.7066551, -0.8882772, 0.6893501

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0936158, upper bound: 3.0463735
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.1506878, 0.0907005, -0.3466392, 0.4922633
1: -0.3946697, 0.4863652, -0.2101478, 0.1401098, -0.5347794, 0.6965131
2: -0.2981569, 0.5062003, -0.1937296, 0.1382380, -0.4363949, 0.6999299
3: -0.5855743, 0.6284050, -0.2145611, 0.1737585, -0.7593328, 0.8429661
4: -0.3760676, 0.6415557, -0.1262522, 0.1747571, -0.5508246, 0.7678078

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.1706024, 0.1344135, -0.3903522, 0.5121780
1: -0.3946697, 0.4863652, -0.2284564, 0.1925970, -0.5872667, 0.7148216
2: -0.2981569, 0.5062003, -0.2065775, 0.1987212, -0.4968781, 0.7127777
3: -0.5855743, 0.6284050, -0.2968557, 0.2297280, -0.8153024, 0.9252607
4: -0.3760676, 0.6415557, -0.1908983, 0.2396057, -0.6156732, 0.8324539

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.1506878, 0.0907005, -0.4994065, 0.7154528
1: -0.6539204, 0.7966581, -0.2101478, 0.1401098, -0.7940302, 1.0068059
2: -0.4457314, 0.8629469, -0.1937296, 0.1382380, -0.5839695, 1.0566764
3: -1.0570911, 1.0608249, -0.2145611, 0.1737585, -1.2308495, 1.2753860
4: -0.6360155, 1.1408229, -0.1262522, 0.1747571, -0.8107727, 1.2670751

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.1706024, 0.1344135, -0.5431195, 0.7353674
1: -0.6539204, 0.7966581, -0.2284564, 0.1925970, -0.8465173, 1.0251145
2: -0.4457314, 0.8629469, -0.2065775, 0.1987212, -0.6444526, 1.0695243
3: -1.0570911, 1.0608249, -0.2968557, 0.2297280, -1.2868191, 1.3576806
4: -0.6360155, 1.1408229, -0.1908983, 0.2396057, -0.8756212, 1.3317212

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.2424042, 0.3243341, -0.5802728, 0.5839797
1: -0.3946697, 0.4863652, -0.3773183, 0.4599679, -0.8546375, 0.8636836
2: -0.2981569, 0.5062003, -0.2913593, 0.4575135, -0.7556703, 0.7975596
3: -0.5855743, 0.6284050, -0.5175009, 0.5963498, -1.1819241, 1.1459059
4: -0.3760676, 0.6415557, -0.3589755, 0.5735825, -0.9496501, 1.0005312

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2559387, 0.3415756, -0.3695980, 0.4989170, -0.7548556, 0.7111736
1: -0.3946697, 0.4863652, -0.6057562, 0.7350104, -1.1296802, 1.0921214
2: -0.2981569, 0.5062003, -0.4243644, 0.6771514, -0.9753083, 0.9305648
3: -0.5855743, 0.6284050, -0.8122164, 0.9872086, -1.5727830, 1.4406214
4: -0.3760676, 0.6415557, -0.5941179, 0.8604028, -1.2364705, 1.2356737

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.2424042, 0.3243341, -0.7330402, 0.8071691
1: -0.6539204, 0.7966581, -0.3773183, 0.4599679, -1.1138883, 1.1739764
2: -0.4457314, 0.8629469, -0.2913593, 0.4575135, -0.9032449, 1.1543062
3: -1.0570911, 1.0608249, -0.5175009, 0.5963498, -1.6534407, 1.5783257
4: -0.6360155, 1.1408229, -0.3589755, 0.5735825, -1.2095981, 1.4997983

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4087062, 0.5647650, -0.3695980, 0.4989170, -0.9076232, 0.9343630
1: -0.6539204, 0.7966581, -0.6057562, 0.7350104, -1.3889307, 1.4024143
2: -0.4457314, 0.8629469, -0.4243644, 0.6771514, -1.1228828, 1.2873113
3: -1.0570911, 1.0608249, -0.8122164, 0.9872086, -2.0442996, 1.8730413
4: -0.6360155, 1.1408229, -0.5941179, 0.8604028, -1.4964184, 1.7349408

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1506878, 0.0907005, -0.4108474, 0.5739069, -0.7245947, 0.5015479
1: -0.2101478, 0.1401098, -0.6647576, 0.7722933, -0.9824411, 0.8048673
2: -0.1937296, 0.1382380, -0.4539672, 0.8914004, -1.0851300, 0.5922053
3: -0.2145611, 0.1737585, -1.0951042, 1.0287758, -1.2433369, 1.2688626
4: -0.1262522, 0.1747571, -0.6342691, 1.2004530, -1.3267052, 0.8090261

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3420067, upper bound: 3.3043490
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3392387, upper bound: 3.2888770
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1506878, 0.0907005, -0.7399998, 1.1651452, -1.3158330, 0.8307002
1: -0.2101478, 0.1401098, -1.1922561, 1.6601820, -1.8703299, 1.3323658
2: -0.1937296, 0.1382380, -0.7848589, 1.7891968, -1.9829264, 0.9230969
3: -0.2145611, 0.1737585, -2.1252515, 2.1598887, -2.3744500, 2.2990098
4: -0.1262522, 0.1747571, -1.2122062, 2.3479633, -2.4742155, 1.3869632

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3420068, upper bound: 3.3105595
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3392386, upper bound: 3.2909828
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1706024, 0.1344135, -0.4108474, 0.5739069, -0.7445093, 0.5452610
1: -0.2284564, 0.1925970, -0.6647576, 0.7722933, -1.0007497, 0.8573546
2: -0.2065775, 0.1987212, -0.4539672, 0.8914004, -1.0979779, 0.6526884
3: -0.2968557, 0.2297280, -1.0951042, 1.0287758, -1.3256315, 1.3248322
4: -0.1908983, 0.2396057, -0.6342691, 1.2004530, -1.3913513, 0.8738747

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213796, upper bound: 3.2984893
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1706024, 0.1344135, -0.7399998, 1.1651452, -1.3357476, 0.8744133
1: -0.2284564, 0.1925970, -1.1922561, 1.6601820, -1.8886384, 1.3848530
2: -0.2065775, 0.1987212, -0.7848589, 1.7891968, -1.9957743, 0.9835801
3: -0.2968557, 0.2297280, -2.1252515, 2.1598887, -2.4567444, 2.3549795
4: -0.1908983, 0.2396057, -1.2122062, 2.3479633, -2.5388615, 1.4518118

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213796, upper bound: 3.3131550
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1442973, 0.0641716, -0.4633702, 0.6546778, -0.7989751, 0.5275418
1: -0.1991781, 0.1041099, -0.7499810, 0.9026212, -1.1017992, 0.8540908
2: -0.1851572, 0.0952137, -0.5000976, 1.0105180, -1.1956751, 0.5953113
3: -0.1959259, 0.1303892, -1.2505035, 1.1993570, -1.3952830, 1.3808928
4: -0.1161131, 0.1196624, -0.7164865, 1.3536885, -1.4698017, 0.8361489

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3363772, upper bound: 3.2902865
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1442973, 0.0641716, -0.9960755, 1.7575388, -1.9018362, 1.0602472
1: -0.1991781, 0.1041099, -1.5950271, 2.5340419, -2.7332201, 1.6991370
2: -0.1851572, 0.0952137, -1.0632966, 2.5719559, -2.7571132, 1.1585102
3: -0.1959259, 0.1303892, -2.8512359, 3.2199693, -3.4158952, 2.9816251
4: -0.1161131, 0.1196624, -1.6742764, 3.3460999, -3.4622130, 1.7939388

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3363771, upper bound: 3.2934276
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1469732, 0.0585246, -0.3748924, 0.5203760, -0.6673492, 0.4334168
1: -0.2018635, 0.0926680, -0.6071245, 0.7133524, -0.9152158, 0.6997923
2: -0.1884660, 0.0827792, -0.4148059, 0.7982628, -0.9867288, 0.4975851
3: -0.1947707, 0.1182943, -0.9749513, 0.9446180, -1.1393887, 1.0932455
4: -0.1158141, 0.1031256, -0.5678948, 1.0561506, -1.1719646, 0.6710204

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3339772, upper bound: 3.2812828
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1469732, 0.0585246, -0.8181107, 1.2976344, -1.4446076, 0.8766351
1: -0.2018635, 0.0926680, -1.3141755, 1.8839147, -2.0857782, 1.4068433
2: -0.1884660, 0.0827792, -0.8629736, 2.0162628, -2.2047288, 0.9457527
3: -0.1947707, 0.1182943, -2.3894250, 2.4515736, -2.6463444, 2.5077193
4: -0.1158141, 0.1031256, -1.3686197, 2.6507945, -2.7666087, 1.4717453

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3339772, upper bound: 3.2812828
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1850611, 0.2096545, -0.2211066, 0.2775607, -0.4626218, 0.4307611
1: -0.2660490, 0.2913542, -0.3221700, 0.3887413, -0.6547903, 0.6135242
2: -0.2290718, 0.3008571, -0.2730529, 0.4024681, -0.6315399, 0.5739100
3: -0.3428191, 0.3601812, -0.4342754, 0.4909782, -0.8337973, 0.7944567
4: -0.2267372, 0.3690930, -0.2787128, 0.4985040, -0.7252413, 0.6478058

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3479636, upper bound: 3.3083296
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3182990
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707116, upper bound: 3.3178212
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1850611, 0.2096545, -0.6227801, 0.8648329, -1.0498940, 0.8324345
1: -0.2660490, 0.2913542, -0.9949624, 1.2503651, -1.5164142, 1.2863165
2: -0.2290718, 0.3008571, -0.6539169, 1.3625121, -1.5915840, 0.9547737
3: -0.3428191, 0.3601812, -1.6873388, 1.6740556, -2.0168748, 2.0475202
4: -0.2267372, 0.3690930, -0.9924154, 1.8411534, -2.0678906, 1.3615084

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3479636, upper bound: 3.3083296
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3286713
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3260525
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3088257, 0.4204044, -0.2211066, 0.2775607, -0.5863864, 0.6415110
1: -0.4969721, 0.6138726, -0.3221700, 0.3887413, -0.8857133, 0.9360427
2: -0.3497591, 0.6176917, -0.2730529, 0.4024681, -0.7522272, 0.8907446
3: -0.7490399, 0.8086340, -0.4342754, 0.4909782, -1.2400180, 1.2429094
4: -0.4839897, 0.8007323, -0.2787128, 0.4985040, -0.9824935, 1.0794451

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208414, upper bound: 3.2386829
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2683777
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3088257, 0.4204044, -0.6227801, 0.8648329, -1.1736586, 1.0431845
1: -0.4969721, 0.6138726, -0.9949624, 1.2503651, -1.7473372, 1.6088350
2: -0.3497591, 0.6176917, -0.6539169, 1.3625121, -1.7122711, 1.2716085
3: -0.7490399, 0.8086340, -1.6873388, 1.6740556, -2.4230950, 2.4959729
4: -0.4839897, 0.8007323, -0.9924154, 1.8411534, -2.3251426, 1.7931477

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208414, upper bound: 3.2744858
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3443260, upper bound: 3.3182276
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1761130, 0.1789525, -0.5490614, 0.8109142, -0.9870272, 0.7280139
1: -0.2476897, 0.2460892, -0.9102525, 1.1428716, -1.3905613, 1.1563416
2: -0.2157476, 0.2579188, -0.6008223, 1.2489667, -1.4647143, 0.8587410
3: -0.3176915, 0.2980479, -1.5505388, 1.5123341, -1.8300257, 1.8485866
4: -0.2045093, 0.3142227, -0.8959278, 1.6582773, -1.8627865, 1.2101505

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1850611, 0.2096545, -0.3633131, 0.4966332, -0.6816943, 0.5729675
1: -0.2660490, 0.2913542, -0.6023729, 0.6898721, -0.9559212, 0.8937271
2: -0.2290718, 0.3008571, -0.4126769, 0.7484713, -0.9775431, 0.7135339
3: -0.3428191, 0.3601812, -0.9434792, 0.9153897, -1.2582088, 1.3036604
4: -0.2267372, 0.3690930, -0.5620179, 0.9950860, -1.2218232, 0.9311109

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3077167, upper bound: 3.3115122
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120791, upper bound: 3.3110587
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2785032, 0.3774387, -0.5490614, 0.8109142, -1.0894173, 0.9265001
1: -0.4475186, 0.5532678, -0.9102525, 1.1428716, -1.5903902, 1.4635203
2: -0.3206336, 0.5484278, -0.6008223, 1.2489667, -1.5696002, 1.1492501
3: -0.6571027, 0.7252599, -1.5505388, 1.5123341, -2.1694369, 2.2757986
4: -0.4330885, 0.7033490, -0.8959278, 1.6582773, -2.0913658, 1.5992768

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3088257, 0.4204044, -0.3633131, 0.4966332, -0.8054589, 0.7837175
1: -0.4969721, 0.6138726, -0.6023729, 0.6898721, -1.1868442, 1.2162455
2: -0.3497591, 0.6176917, -0.4126769, 0.7484713, -1.0982304, 1.0303684
3: -0.7490399, 0.8086340, -0.9434792, 0.9153897, -1.6644297, 1.7521132
4: -0.4839897, 0.8007323, -0.5620179, 0.9950860, -1.4790756, 1.3627501

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2920542, upper bound: 3.2594920
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119140, upper bound: 3.3048480
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.0676768, 0.0203616, -0.2414682, 0.3452376
1: -0.3221700, 0.3887413, -0.0773866, 0.0391565, -0.3613265, 0.4661279
2: -0.2730529, 0.4024681, -0.0644345, 0.0282484, -0.3013013, 0.4669026
3: -0.4342754, 0.4909782, -0.0712000, 0.0471544, -0.4814299, 0.5621781
4: -0.2787128, 0.4985040, -0.0532835, 0.0354563, -0.3141691, 0.5517876

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1927976, upper bound: 3.2837398
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1922949, upper bound: 3.2818812
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.1649871, 0.1174907, -0.3385974, 0.4425478
1: -0.3221700, 0.3887413, -0.2215959, 0.1765458, -0.4987158, 0.6103371
2: -0.2730529, 0.4024681, -0.1996021, 0.1877475, -0.4608004, 0.6020702
3: -0.4342754, 0.4909782, -0.2914082, 0.2083298, -0.6426053, 0.7823864
4: -0.2787128, 0.4985040, -0.1816220, 0.2294925, -0.5082053, 0.6801261

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1927976, upper bound: 3.2837398
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1922949, upper bound: 3.2818812
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.6052738, 0.8398099, -0.0676768, 0.0203616, -0.6256353, 0.9074867
1: -0.9567434, 1.2019296, -0.0773866, 0.0391565, -0.9958999, 1.2793162
2: -0.6341691, 1.3133148, -0.0644345, 0.0282484, -0.6624175, 1.3777493
3: -1.6088847, 1.6106904, -0.0712000, 0.0471544, -1.6560391, 1.6818904
4: -0.9545741, 1.7640181, -0.0532835, 0.0354563, -0.9900302, 1.8173016

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918414, upper bound: 3.2801345
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1922239, upper bound: 3.2814532
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.6052738, 0.8398099, -0.1649871, 0.1174907, -0.7227646, 1.0047970
1: -0.9567434, 1.2019296, -0.2215959, 0.1765458, -1.1332892, 1.4235255
2: -0.6341691, 1.3133148, -0.1996021, 0.1877475, -0.8219166, 1.5129169
3: -1.6088847, 1.6106904, -0.2914082, 0.2083298, -1.8172145, 1.9020985
4: -0.9545741, 1.7640181, -0.1816220, 0.2294925, -1.1840665, 1.9456401

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918414, upper bound: 3.2801345
time: 0.28 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1922239, upper bound: 3.2814532
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.1842154, 0.2073062, -0.4284129, 0.4617761
1: -0.3221700, 0.3887413, -0.2644064, 0.2852268, -0.6073968, 0.6531476
2: -0.2730529, 0.4024681, -0.2280039, 0.2978324, -0.5708853, 0.6304720
3: -0.4342754, 0.4909782, -0.3391942, 0.3525377, -0.7868131, 0.8301724
4: -0.2787128, 0.4985040, -0.2241528, 0.3658484, -0.6445613, 0.7226568

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2667848, upper bound: 3.3450416
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2683777, upper bound: 3.3443260
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2211066, 0.2775607, -0.2918625, 0.3971892, -0.6182958, 0.5694232
1: -0.3221700, 0.3887413, -0.4734699, 0.5811095, -0.9032795, 0.8622112
2: -0.2730529, 0.4024681, -0.3393690, 0.5553786, -0.8284315, 0.7418371
3: -0.4342754, 0.4909782, -0.6585814, 0.7684231, -1.2026985, 1.1495596
4: -0.2787128, 0.4985040, -0.4620743, 0.7093326, -0.9880455, 0.9605784

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2667848, upper bound: 3.3450416
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2683777, upper bound: 3.3443260
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6227801, 0.8648329, -0.1842154, 0.2073062, -0.8300864, 1.0490483
1: -0.9949624, 1.2503651, -0.2644064, 0.2852268, -1.2801893, 1.5147715
2: -0.6539169, 1.3625121, -0.2280039, 0.2978324, -0.9517491, 1.5905160
3: -1.6873388, 1.6740556, -0.3391942, 0.3525377, -2.0398765, 2.0132499
4: -0.9924154, 1.8411534, -0.2241528, 0.3658484, -1.3582639, 2.0653062

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2792289, upper bound: 3.3465769
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3162346, upper bound: 3.3555851
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6227801, 0.8648329, -0.2918625, 0.3971892, -1.0199692, 1.1566954
1: -0.9949624, 1.2503651, -0.4734699, 0.5811095, -1.5760719, 1.7238351
2: -0.6539169, 1.3625121, -0.3393690, 0.5553786, -1.2092952, 1.7018812
3: -1.6873388, 1.6740556, -0.6585814, 0.7684231, -2.4557619, 2.3326371
4: -0.9924154, 1.8411534, -0.4620743, 0.7093326, -1.7017480, 2.3032277

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2792289, upper bound: 3.3465769
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3162346, upper bound: 3.3555851
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3456418, 0.4810718, -0.3449760, 0.4804536, -0.8260953, 0.8260478
1: -0.5610024, 0.6868975, -0.5595652, 0.6852832, -1.2462857, 1.2464627
2: -0.3901013, 0.7285742, -0.3894793, 0.7272860, -1.1173873, 1.1180534
3: -0.8672440, 0.8995361, -0.8643408, 0.8972304, -1.7644744, 1.7638768
4: -0.5257008, 0.9459569, -0.5243616, 0.9438926, -1.4695934, 1.4703183

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3535225, upper bound: 3.3831168
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3534328, upper bound: 3.3810104
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3456418, 0.4810718, -0.2695478, 0.3760729, -0.7217146, 0.7506194
1: -0.5610024, 0.6868975, -0.4326958, 0.5375808, -1.0985832, 1.1195933
2: -0.3901013, 0.7285742, -0.3200988, 0.5548421, -0.9449434, 1.0486729
3: -0.8672440, 0.8995361, -0.6285393, 0.6923354, -1.5595794, 1.5280752
4: -0.5257008, 0.9459569, -0.3963178, 0.7018194, -1.2275202, 1.3422747

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3535225, upper bound: 3.3831168
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3534328, upper bound: 3.3810104
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2695478, 0.3760729, -0.3456418, 0.4810718, -0.7506195, 0.7217146
1: -0.4326958, 0.5375808, -0.5610024, 0.6868975, -1.1195933, 1.0985832
2: -0.3200988, 0.5548421, -0.3901013, 0.7285742, -1.0486729, 0.9449434
3: -0.6285393, 0.6923354, -0.8672440, 0.8995361, -1.5280752, 1.5595794
4: -0.3963178, 0.7018194, -0.5257008, 0.9459569, -1.3422747, 1.2275202

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3334450, upper bound: 3.3375030
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2695478, 0.3760729, -0.2695478, 0.3760729, -0.6456205, 0.6456205
1: -0.4326958, 0.5375808, -0.4326958, 0.5375808, -0.9702767, 0.9702767
2: -0.3200988, 0.5548421, -0.3200988, 0.5548421, -0.8749408, 0.8749407
3: -0.6285393, 0.6923354, -0.6285393, 0.6923354, -1.3208747, 1.3208747
4: -0.3963178, 0.7018194, -0.3963178, 0.7018194, -1.0981371, 1.0981371

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3334450, upper bound: 3.3375030
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3456418, 0.4810718, -0.3730532, 0.5142136, -0.8598554, 0.8541250
1: -0.5610024, 0.6868975, -0.6110075, 0.6962433, -1.2572458, 1.2979050
2: -0.3901013, 0.7285742, -0.4200994, 0.7856184, -1.1757197, 1.1486735
3: -0.8672440, 0.8995361, -0.9770764, 0.9289197, -1.7961638, 1.8766119
4: -0.5257008, 0.9459569, -0.5794039, 1.0543439, -1.5800447, 1.5253605

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132309, upper bound: 3.3540007
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3225055, upper bound: 3.3548954
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2697954, 0.3749946, -0.3664802, 0.5043814, -0.7741768, 0.7414748
1: -0.4367954, 0.5320600, -0.6070046, 0.6854192, -1.1222146, 1.1390646
2: -0.3219306, 0.5583752, -0.4137102, 0.7665520, -1.0884826, 0.9720851
3: -0.6400712, 0.6883482, -0.9603794, 0.9151838, -1.5552551, 1.6487274
4: -0.3984006, 0.7076451, -0.5638500, 1.0202947, -1.4186952, 1.2714950

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3502041
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2695478, 0.3760729, -0.4467238, 0.6215951, -0.8911428, 0.8227966
1: -0.4326958, 0.5375808, -0.7328915, 0.8427354, -1.2754313, 1.2704720
2: -0.3200988, 0.5548421, -0.4924225, 0.9656458, -1.2857447, 1.0472646
3: -0.6285393, 0.6923354, -1.2085794, 1.1289421, -1.7574815, 1.9009148
4: -0.3963178, 0.7018194, -0.7015048, 1.3040695, -1.7003874, 1.4033240

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2092264, 0.2504458, -0.4117214, 0.5673239, -0.7765503, 0.6621672
1: -0.3016140, 0.3417601, -0.6853573, 0.7768623, -1.0784762, 1.0271173
2: -0.2575050, 0.3613721, -0.4597958, 0.8666953, -1.1242003, 0.8211678
3: -0.3914973, 0.4247779, -1.0987918, 1.0422142, -1.4337115, 1.5235697
4: -0.2490113, 0.4423591, -0.6412443, 1.1624334, -1.4114447, 1.0836034

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3730532, 0.5142136, -0.3456418, 0.4810718, -0.8541249, 0.8598554
1: -0.6110075, 0.6962433, -0.5610024, 0.6868975, -1.2979050, 1.2572458
2: -0.4200994, 0.7856184, -0.3901013, 0.7285742, -1.1486735, 1.1757197
3: -0.9770764, 0.9289197, -0.8672440, 0.8995361, -1.8766125, 1.7961634
4: -0.5794039, 1.0543439, -0.5257008, 0.9459569, -1.5253605, 1.5800447

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4467238, 0.6215951, -0.2695478, 0.3760729, -0.8227966, 0.8911428
1: -0.7328915, 0.8427354, -0.4326958, 0.5375808, -1.2704720, 1.2754313
2: -0.4924225, 0.9656458, -0.3200988, 0.5548421, -1.0472646, 1.2857447
3: -1.2085794, 1.1289421, -0.6285393, 0.6923354, -1.9009148, 1.7574815
4: -0.7015048, 1.3040695, -0.3963178, 0.7018194, -1.4033240, 1.7003874

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3664802, 0.5043814, -0.2697954, 0.3749946, -0.7414748, 0.7741768
1: -0.6070046, 0.6854192, -0.4367954, 0.5320600, -1.1390644, 1.1222146
2: -0.4137102, 0.7665520, -0.3219306, 0.5583752, -0.9720852, 1.0884826
3: -0.9603794, 0.9151838, -0.6400712, 0.6883482, -1.6487277, 1.5552551
4: -0.5638500, 1.0202947, -0.3984006, 0.7076451, -1.2714950, 1.4186952

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4117214, 0.5673239, -0.2092264, 0.2504458, -0.6621672, 0.7765503
1: -0.6853573, 0.7768623, -0.3016140, 0.3417601, -1.0271174, 1.0784762
2: -0.4597958, 0.8666953, -0.2575050, 0.3613721, -0.8211678, 1.1242003
3: -1.0987918, 1.0422142, -0.3914973, 0.4247779, -1.5235697, 1.4337115
4: -0.6412443, 1.1624334, -0.2490113, 0.4423591, -1.0836034, 1.4114447

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4467238, 0.6215951, -0.4467238, 0.6215951, -1.0683187, 1.0683186
1: -0.7328915, 0.8427354, -0.7328915, 0.8427354, -1.5756269, 1.5756269
2: -0.4924225, 0.9656458, -0.4924225, 0.9656458, -1.4580684, 1.4580684
3: -1.2085794, 1.1289421, -1.2085794, 1.1289421, -2.3375216, 2.3375216
4: -0.7015048, 1.3040695, -0.7015048, 1.3040695, -2.0055742, 2.0055742

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124995, upper bound: 3.3130727
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4467238, 0.6215951, -0.4117214, 0.5673239, -1.0140475, 1.0333165
1: -0.7328915, 0.8427354, -0.6853573, 0.7768623, -1.5097535, 1.5280926
2: -0.4924225, 0.9656458, -0.4597958, 0.8666953, -1.3591177, 1.4254416
3: -1.2085794, 1.1289421, -1.0987918, 1.0422142, -2.2507935, 2.2277341
4: -0.7015048, 1.3040695, -0.6412443, 1.1624334, -1.8639379, 1.9453138

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124995, upper bound: 3.3130727
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3664802, 0.5043814, -0.6279464, 0.9307750, -1.2972553, 1.1323278
1: -0.6070046, 0.6854192, -1.0247976, 1.3076565, -1.9146612, 1.7102165
2: -0.4137102, 0.7665520, -0.6732014, 1.4444041, -1.8581141, 1.4397533
3: -0.9603794, 0.9151838, -1.7774800, 1.7267673, -2.6871464, 2.6926637
4: -0.5638500, 1.0202947, -1.0197093, 1.9138947, -2.4777443, 2.0400040

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4117214, 0.5673239, -0.3747006, 0.5153155, -0.9270369, 0.9420245
1: -0.6853573, 0.7768623, -0.6229787, 0.7066517, -1.3920089, 1.3998410
2: -0.4597958, 0.8666953, -0.4230216, 0.7808440, -1.2406398, 1.2897170
3: -1.0987918, 1.0422142, -0.9837797, 0.9448279, -2.0436192, 2.0259938
4: -0.6412443, 1.1624334, -0.5836946, 1.0469630, -1.6882073, 1.7461276

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.01 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1498572, upper bound: 3.2329620
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1498572, upper bound: 3.2639925
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0939294, upper bound: 3.0463998
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0939294, upper bound: 3.0463998
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1443273, upper bound: 3.2233479
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2503868, upper bound: 3.2938263
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1443273, upper bound: 3.2233479
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2503868, upper bound: 3.2938263
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0936158, upper bound: 3.0463735
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.0936158, upper bound: 3.0463735
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2229522, upper bound: 3.1772977
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3420067, upper bound: 3.3043490
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3392387, upper bound: 3.2888770
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3420068, upper bound: 3.3105595
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3392386, upper bound: 3.2909828
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3182990
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3707116, upper bound: 3.3178212
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3286713
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3260525
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3208414, upper bound: 3.2386829
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2683777
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3208414, upper bound: 3.2744858
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3443260, upper bound: 3.3182276
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3077167, upper bound: 3.3115122
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3120791, upper bound: 3.3110587
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2920542, upper bound: 3.2594920
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3119140, upper bound: 3.3048480
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1927976, upper bound: 3.2837398
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1922949, upper bound: 3.2818812
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1927976, upper bound: 3.2837398
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1922949, upper bound: 3.2818812
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1918414, upper bound: 3.2801345
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1922239, upper bound: 3.2814532
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1918414, upper bound: 3.2801345
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.1922239, upper bound: 3.2814532
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2667848, upper bound: 3.3450416
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2683777, upper bound: 3.3443260
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2667848, upper bound: 3.3450416
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2683777, upper bound: 3.3443260
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2792289, upper bound: 3.3465769
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3162346, upper bound: 3.3555851
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.2792289, upper bound: 3.3465769
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3162346, upper bound: 3.3555851
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3535225, upper bound: 3.3831168
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3534328, upper bound: 3.3810104
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3535225, upper bound: 3.3831168
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3534328, upper bound: 3.3810104
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3334450, upper bound: 3.3375030
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3334450, upper bound: 3.3375030
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3132309, upper bound: 3.3540007
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3225055, upper bound: 3.3548954
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3502041
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3194967, upper bound: 3.3137272
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3124995, upper bound: 3.3130727
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3124995, upper bound: 3.3130727
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.1520702, 0.0857642, -0.1482848, 0.1537314
1: -0.0692971, 0.0076117, -0.1991930, 0.1328123, -0.2021094, 0.2068048
2: -0.0580488, 0.0014766, -0.1822435, 0.1380647, -0.1961136, 0.1837200
3: -0.0626092, 0.0125836, -0.2571985, 0.1591592, -0.2217683, 0.2697821
4: -0.0435550, 0.0022268, -0.1675957, 0.1625305, -0.2060855, 0.1698225

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0664968, 0.0116788, -0.1520702, 0.0857642, -0.1522611, 0.1637490
1: -0.0752405, 0.0252749, -0.1991930, 0.1328123, -0.2080528, 0.2244680
2: -0.0628744, 0.0152150, -0.1822435, 0.1380647, -0.2009391, 0.1974585
3: -0.0685259, 0.0332320, -0.2571985, 0.1591592, -0.2276851, 0.2904304
4: -0.0497595, 0.0198232, -0.1675957, 0.1625305, -0.2122900, 0.1874189

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.1718930, 0.1522878, -0.2148083, 0.1735542
1: -0.0692971, 0.0076117, -0.2361045, 0.2212102, -0.2905073, 0.2437163
2: -0.0580488, 0.0014766, -0.2093908, 0.2357066, -0.2937555, 0.2108674
3: -0.0626092, 0.0125836, -0.3105830, 0.2643077, -0.3269168, 0.3231666
4: -0.0435550, 0.0022268, -0.1935923, 0.2920630, -0.3356180, 0.1958191

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2329620
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2329620
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0666025, 0.0125595, -0.1718930, 0.1522878, -0.2188902, 0.1844525
1: -0.0755262, 0.0264283, -0.2361045, 0.2212102, -0.2967364, 0.2625329
2: -0.0630669, 0.0163602, -0.2093908, 0.2357066, -0.2987736, 0.2257510
3: -0.0687606, 0.0340288, -0.3105830, 0.2643077, -0.3330683, 0.3446118
4: -0.0499745, 0.0211048, -0.1935923, 0.2920630, -0.3420375, 0.2146971

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2581693
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2581693
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.0652147, 0.0083182, -0.1196646, 0.1019372
1: -0.1613708, 0.0607112, -0.0734224, 0.0195891, -0.1809599, 0.1341335
2: -0.1328885, 0.0555079, -0.0614410, 0.0105155, -0.1434040, 0.1169489
3: -0.1744612, 0.0704553, -0.0668020, 0.0271159, -0.2015771, 0.1372573
4: -0.0577233, 0.0723308, -0.0479105, 0.0142335, -0.0719568, 0.1202412

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1357337, upper bound: 3.0607899
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1357337, upper bound: 3.0610602
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.0676768, 0.0203616, -0.1800252, 0.1617803
1: -0.2106493, 0.1473507, -0.0773866, 0.0391565, -0.2498058, 0.2247373
2: -0.1921865, 0.1520855, -0.0644345, 0.0282484, -0.2204349, 0.2165200
3: -0.2745035, 0.1743023, -0.0712000, 0.0471544, -0.3216580, 0.2455022
4: -0.1750863, 0.1810839, -0.0532835, 0.0354563, -0.2105426, 0.2343674

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2329620, upper bound: 3.1498572
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2329620, upper bound: 3.1864269
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.1448865, 0.0710705, -0.1824169, 0.1816090
1: -0.1613708, 0.0607112, -0.2009955, 0.1135176, -0.2748884, 0.2617067
2: -0.1328885, 0.0555079, -0.1859161, 0.1115460, -0.2444345, 0.2414240
3: -0.1744612, 0.0704553, -0.2007326, 0.1407833, -0.3152445, 0.2711880
4: -0.0577233, 0.0723308, -0.1176714, 0.1355742, -0.1932974, 0.1900021

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0162219, upper bound: 3.0162219
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0162219, upper bound: 3.0463998
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.1649871, 0.1174907, -0.2771544, 0.2590906
1: -0.2106493, 0.1473507, -0.2215959, 0.1765458, -0.3871951, 0.3689465
2: -0.1921865, 0.1520855, -0.1996021, 0.1877475, -0.3799340, 0.3516876
3: -0.2745035, 0.1743023, -0.2914082, 0.2083298, -0.4828334, 0.4657105
4: -0.1750863, 0.1810839, -0.1816220, 0.2294925, -0.4045787, 0.3627059

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0463998, upper bound: 3.0939294
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0463998, upper bound: 3.1566236
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.1741959, 0.1748457, -0.2373662, 0.1758571
1: -0.0692971, 0.0076117, -0.2434373, 0.2397725, -0.3090696, 0.2510490
2: -0.0580488, 0.0014766, -0.2128446, 0.2528849, -0.3109337, 0.2143212
3: -0.0626092, 0.0125836, -0.3126002, 0.2900882, -0.3526974, 0.3251839
4: -0.0435550, 0.0022268, -0.2017988, 0.3083061, -0.3518611, 0.2040256

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0666025, 0.0125595, -0.1842154, 0.2073062, -0.2739087, 0.1967749
1: -0.0755262, 0.0264283, -0.2644064, 0.2852268, -0.3607530, 0.2908347
2: -0.0630669, 0.0163602, -0.2280039, 0.2978324, -0.3608993, 0.2443641
3: -0.0687606, 0.0340288, -0.3391942, 0.3525377, -0.4212983, 0.3732231
4: -0.0499745, 0.0211048, -0.2241528, 0.3658484, -0.4158229, 0.2452576

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2953006, upper bound: 3.2812533
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2813166
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.2395068, 0.3274221, -0.3899426, 0.2411680
1: -0.0692971, 0.0076117, -0.3823066, 0.4701446, -0.5394418, 0.3899184
2: -0.0580488, 0.0014766, -0.2868920, 0.4599918, -0.5180406, 0.2883686
3: -0.0626092, 0.0125836, -0.5227154, 0.6138681, -0.6764772, 0.5352991
4: -0.0435550, 0.0022268, -0.3670687, 0.5786202, -0.6221752, 0.3692956

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0666025, 0.0125595, -0.2910421, 0.3949972, -0.4615997, 0.3036016
1: -0.0755262, 0.0264283, -0.4709392, 0.5766153, -0.6521415, 0.4973675
2: -0.0630669, 0.0163602, -0.3379044, 0.5531360, -0.6162029, 0.3542646
3: -0.0687606, 0.0340288, -0.6551671, 0.7631698, -0.8319305, 0.6891959
4: -0.0499745, 0.0211048, -0.4598577, 0.7066551, -0.7566296, 0.4809625

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2221127, upper bound: 3.2601534
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2489007, upper bound: 3.2678566
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.1741959, 0.1748457, -0.2861921, 0.2109185
1: -0.1613708, 0.0607112, -0.2434373, 0.2397725, -0.4011433, 0.3041485
2: -0.1328885, 0.0555079, -0.2128446, 0.2528849, -0.3857734, 0.2683524
3: -0.1744612, 0.0704553, -0.3126002, 0.2900882, -0.4645494, 0.3830555
4: -0.0577233, 0.0723308, -0.2017988, 0.3083061, -0.3660294, 0.2741296

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.1842154, 0.2073062, -0.3669699, 0.2783189
1: -0.2106493, 0.1473507, -0.2644064, 0.2852268, -0.4958761, 0.4117570
2: -0.1921865, 0.1520855, -0.2280039, 0.2978324, -0.4900188, 0.3800894
3: -0.2745035, 0.1743023, -0.3391942, 0.3525377, -0.6270412, 0.5134965
4: -0.1750863, 0.1810839, -0.2241528, 0.3658484, -0.5409347, 0.4052367

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2586818, upper bound: 3.1847512
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1849856
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.2395068, 0.3274221, -0.4387685, 0.2762294
1: -0.1613708, 0.0607112, -0.3823066, 0.4701446, -0.6315154, 0.4430178
2: -0.1328885, 0.0555079, -0.2868920, 0.4599918, -0.5928802, 0.3423999
3: -0.1744612, 0.0704553, -0.5227154, 0.6138681, -0.7883292, 0.5931708
4: -0.0577233, 0.0723308, -0.3670687, 0.5786202, -0.6363435, 0.4393995

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.2910421, 0.3949972, -0.5546609, 0.3851456
1: -0.2106493, 0.1473507, -0.4709392, 0.5766153, -0.7872646, 0.6182898
2: -0.1921865, 0.1520855, -0.3379044, 0.5531360, -0.7453225, 0.4899899
3: -0.2745035, 0.1743023, -0.6551671, 0.7631698, -1.0376734, 0.8294693
4: -0.1750863, 0.1810839, -0.4598577, 0.7066551, -0.8817414, 0.6409416

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8137068, upper bound: 2.8203866
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2224935, upper bound: 3.1769383
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1479754, 0.0622823, -0.3895089, 0.5431411, -0.6911165, 0.4517912
1: -0.2030102, 0.1032830, -0.6299436, 0.7307837, -0.9337939, 0.7332265
2: -0.1896725, 0.0943340, -0.4335223, 0.8404056, -1.0300781, 0.5278563
3: -0.1976196, 0.1311589, -1.0278971, 0.9717065, -1.1693261, 1.1590559
4: -0.1192789, 0.1223733, -0.5994228, 1.1278505, -1.2471294, 0.7217961

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3416311, upper bound: 3.3037967
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3258777, upper bound: 3.3007828
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1530817, 0.1312132, -0.3787739, 0.5277484, -0.6808301, 0.5099871
1: -0.2193033, 0.1904956, -0.6126482, 0.7130275, -0.9323308, 0.8031437
2: -0.1977287, 0.2014404, -0.4237821, 0.8133011, -1.0110297, 0.6252224
3: -0.2352768, 0.2330002, -0.9916837, 0.9466166, -1.1818935, 1.2246838
4: -0.1397993, 0.2402222, -0.5829123, 1.0883541, -1.2281535, 0.8231345

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3390803, upper bound: 3.2884686
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3232250, upper bound: 3.2844933
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1479754, 0.0622823, -0.7159302, 1.1199043, -1.2678797, 0.7782124
1: -0.2030102, 0.1032830, -1.1550657, 1.5947433, -1.7977535, 1.2583486
2: -0.1896725, 0.0943340, -0.7603088, 1.7221472, -1.9118197, 0.8546428
3: -0.1976196, 0.1311589, -2.0520563, 2.0763783, -2.2739980, 2.1832151
4: -0.1192789, 0.1223733, -1.1694230, 2.2626328, -2.3819118, 1.2917961

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2875735
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2909828
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1530817, 0.1312132, -0.7180997, 1.1219785, -1.2750602, 0.8493129
1: -0.2193033, 0.1904956, -1.1574483, 1.5994399, -1.8187431, 1.3479439
2: -0.1977287, 0.2014404, -0.7618335, 1.7259176, -1.9236462, 0.9632739
3: -0.2352768, 0.2330002, -2.0556724, 2.0829406, -2.3182175, 2.2886727
4: -0.1397993, 0.2402222, -1.1738890, 2.2670481, -2.4068475, 1.4141113

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2875735
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2909828
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1712859, 0.1554839, -0.2130781, 0.2555990, -0.4268849, 0.3685620
1: -0.2360464, 0.2119974, -0.3067773, 0.3534110, -0.5894574, 0.5187747
2: -0.2078791, 0.2236603, -0.2626007, 0.3711776, -0.5790567, 0.4862610
3: -0.3012584, 0.2526170, -0.3994018, 0.4414256, -0.7426840, 0.6520188
4: -0.1917115, 0.2698154, -0.2568902, 0.4573742, -0.6490856, 0.5267056

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3178212
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3178212
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2265842, 0.3310158, -0.2146745, 0.2607463, -0.4873305, 0.5456904
1: -0.3268467, 0.4562286, -0.3101411, 0.3619015, -0.6887482, 0.7663696
2: -0.2707975, 0.4664631, -0.2647355, 0.3780656, -0.6488631, 0.7311985
3: -0.4477058, 0.5785184, -0.4039366, 0.4531335, -0.9008393, 0.9824550
4: -0.3149576, 0.5780836, -0.2611167, 0.4662192, -0.7811767, 0.8392003

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3178212
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3178212
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1712859, 0.1554839, -0.6002746, 0.8339846, -1.0052705, 0.7557585
1: -0.2360464, 0.2119974, -0.9594641, 1.2053905, -1.4414368, 1.1714616
2: -0.2078791, 0.2236603, -0.6319249, 1.3117821, -1.5196612, 0.8555851
3: -0.3012584, 0.2526170, -1.6208720, 1.6121056, -1.9133639, 1.8734890
4: -0.1917115, 0.2698154, -0.9543027, 1.7681477, -1.9598593, 1.2241180

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707288, upper bound: 3.3260525
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3707288, upper bound: 3.3260525
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2265842, 0.3310158, -0.6047331, 0.8402957, -1.0668799, 0.9357489
1: -0.3268467, 0.4562286, -0.9659957, 1.2164911, -1.5433378, 1.4222240
2: -0.2707975, 0.4664631, -0.6360962, 1.3209425, -1.5917400, 1.1025592
3: -0.4477058, 0.5785184, -1.6320602, 1.6271042, -2.0748100, 2.2105784
4: -0.3149576, 0.5780836, -0.9632178, 1.7809364, -2.0958939, 1.5413014

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3709153, upper bound: 3.3260525
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3709153, upper bound: 3.3260525
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2468112, 0.3300309, -0.2130781, 0.2555990, -0.5024102, 0.5431090
1: -0.3955103, 0.4865677, -0.3067773, 0.3534110, -0.7489213, 0.7933450
2: -0.2913569, 0.4733481, -0.2626007, 0.3711776, -0.6625345, 0.7359488
3: -0.5598316, 0.6335915, -0.3994018, 0.4414256, -1.0012572, 1.0329933
4: -0.3778501, 0.5984451, -0.2568902, 0.4573742, -0.8352243, 0.8553352

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208409, upper bound: 3.2386825
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208409, upper bound: 3.2386825
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3219413, 0.4448701, -0.2146745, 0.2607463, -0.5826874, 0.6595446
1: -0.5123811, 0.6477650, -0.3101411, 0.3619015, -0.8742826, 0.9579061
2: -0.3597301, 0.6536453, -0.2647355, 0.3780656, -0.7377957, 0.9183808
3: -0.7773622, 0.8556224, -0.4039366, 0.4531335, -1.2304955, 1.2595589
4: -0.5105215, 0.8503320, -0.2611167, 0.4662192, -0.9767407, 1.1114486

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2667848
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2683777
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2468112, 0.3300309, -0.6002746, 0.8339846, -1.0807958, 0.9303055
1: -0.3955103, 0.4865677, -0.9594641, 1.2053905, -1.6009008, 1.4460317
2: -0.2913569, 0.4733481, -0.6319249, 1.3117821, -1.6031389, 1.1052730
3: -0.5598316, 0.6335915, -1.6208720, 1.6121056, -2.1719372, 2.2544632
4: -0.3778501, 0.5984451, -0.9543027, 1.7681477, -2.1459978, 1.5527475

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3516686, upper bound: 3.2744858
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3516686, upper bound: 3.2744858
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3219413, 0.4448701, -0.6047331, 0.8402957, -1.1622368, 1.0496032
1: -0.5123811, 0.6477650, -0.9659957, 1.2164911, -1.7288718, 1.6137607
2: -0.3597301, 0.6536453, -0.6360962, 1.3209425, -1.6806726, 1.2897413
3: -0.7773622, 0.8556224, -1.6320602, 1.6271042, -2.4044664, 2.4876823
4: -0.5105215, 0.8503320, -0.9632178, 1.7809364, -2.2914577, 1.8135498

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3602240, upper bound: 3.3176191
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3602240, upper bound: 3.3182276
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1712859, 0.1554839, -0.3445343, 0.4695941, -0.6408800, 0.5000182
1: -0.2360464, 0.2119974, -0.5692107, 0.6516510, -0.8876974, 0.7812080
2: -0.2078791, 0.2236603, -0.3937661, 0.7057208, -0.9135999, 0.6174263
3: -0.3012584, 0.2526170, -0.8826869, 0.8621734, -1.1634318, 1.1353039
4: -0.1917115, 0.2698154, -0.5292529, 0.9321347, -1.1238462, 0.7990682

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2265842, 0.3310158, -0.3349559, 0.4551466, -0.6817307, 0.6659717
1: -0.3268467, 0.4562286, -0.5515019, 0.6354143, -0.9622610, 1.0077305
2: -0.2707975, 0.4664631, -0.3842540, 0.6807249, -0.9515224, 0.8507171
3: -0.4477058, 0.5785184, -0.8467019, 0.8381586, -1.2858644, 1.4252203
4: -0.3149576, 0.5780836, -0.5127077, 0.8942118, -1.2091694, 1.0907912

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2468112, 0.3300309, -0.3445343, 0.4695941, -0.7164053, 0.6745651
1: -0.3955103, 0.4865677, -0.5692107, 0.6516510, -1.0471613, 1.0557784
2: -0.2913569, 0.4733481, -0.3937661, 0.7057208, -0.9970777, 0.8671142
3: -0.5598316, 0.6335915, -0.8826869, 0.8621734, -1.4220049, 1.5162780
4: -0.3778501, 0.5984451, -0.5292529, 0.9321347, -1.3099848, 1.1276976

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3219413, 0.4448701, -0.3349559, 0.4551466, -0.7770878, 0.7798260
1: -0.5123811, 0.6477650, -0.5515019, 0.6354143, -1.1477952, 1.1992669
2: -0.3597301, 0.6536453, -0.3842540, 0.6807249, -1.0404549, 1.0378993
3: -0.7773622, 0.8556224, -0.8467019, 0.8381586, -1.6155205, 1.7023243
4: -0.5105215, 0.8503320, -0.5127077, 0.8942118, -1.4047333, 1.3630396

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2052934, 0.2239504, -0.0671433, 0.0159121, -0.2212055, 0.2910937
1: -0.2904751, 0.3086712, -0.0763563, 0.0313490, -0.3218241, 0.3850275
2: -0.2511725, 0.3277092, -0.0637104, 0.0215070, -0.2726795, 0.3914196
3: -0.3762597, 0.3788806, -0.0699091, 0.0388068, -0.4150665, 0.4487897
4: -0.2353001, 0.4013989, -0.0509824, 0.0275586, -0.2628587, 0.4523813

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000038, upper bound: 3.3662019
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3098595, 0.4358088, -0.0671590, 0.0166013, -0.3264607, 0.5029678
1: -0.5251927, 0.6571004, -0.0763860, 0.0326574, -0.5578501, 0.7334864
2: -0.3664192, 0.6346077, -0.0637320, 0.0224991, -0.3889183, 0.6983397
3: -0.7393374, 0.8656781, -0.0699868, 0.0401124, -0.7794498, 0.9356648
4: -0.4978842, 0.8078210, -0.0513019, 0.0284835, -0.5263677, 0.8591229

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2052934, 0.2239504, -0.1629100, 0.1079940, -0.3132875, 0.3868604
1: -0.2904751, 0.3086712, -0.2174355, 0.1641184, -0.4545936, 0.5261067
2: -0.2511725, 0.3277092, -0.1967028, 0.1740996, -0.4252721, 0.5244119
3: -0.3762597, 0.3788806, -0.2849931, 0.1934574, -0.5697170, 0.6638736
4: -0.2353001, 0.4013989, -0.1786502, 0.2112573, -0.4465574, 0.5800492

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2769040
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2818812
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3098595, 0.4358088, -0.1627242, 0.1067304, -0.4165899, 0.5985330
1: -0.5251927, 0.6571004, -0.2170347, 0.1627807, -0.6879734, 0.8741351
2: -0.3664192, 0.6346077, -0.1964690, 0.1719500, -0.5383692, 0.8310767
3: -0.7393374, 0.8656781, -0.2843813, 0.1918265, -0.9311639, 1.1500593
4: -0.4978842, 0.8078210, -0.1784653, 0.2082254, -0.7061096, 0.9862863

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2769040
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2818812
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5261998, 0.7323977, -0.0671433, 0.0159121, -0.5421119, 0.7995411
1: -0.8345441, 1.0518881, -0.0763563, 0.0313490, -0.8658930, 1.1282444
2: -0.5579855, 1.1353525, -0.0637104, 0.0215070, -0.5794925, 1.1990629
3: -1.3774799, 1.4013927, -0.0699091, 0.0388068, -1.4162865, 1.4713018
4: -0.8244473, 1.5100366, -0.0509824, 0.0275586, -0.8520058, 1.5610189

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7393270, 1.0187854, -0.0671590, 0.0166013, -0.7559282, 1.0859443
1: -1.1469342, 1.4806118, -0.0763860, 0.0326574, -1.1795913, 1.5569978
2: -0.7548296, 1.6063006, -0.0637320, 0.0224991, -0.7773287, 1.6700325
3: -1.9658682, 1.9879127, -0.0699868, 0.0401124, -2.0059800, 2.0578995
4: -1.1782980, 2.1738334, -0.0513019, 0.0284835, -1.2067815, 2.2251353

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5261998, 0.7323977, -0.1629100, 0.1079940, -0.6341938, 0.8953078
1: -0.8345441, 1.0518881, -0.2174355, 0.1641184, -0.9986625, 1.2693236
2: -0.5579855, 1.1353525, -0.1967028, 0.1740996, -0.7320850, 1.3320553
3: -1.3774799, 1.4013927, -0.2849931, 0.1934574, -1.5709370, 1.6863858
4: -0.8244473, 1.5100366, -0.1786502, 0.2112573, -1.0357046, 1.6886868

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887763, upper bound: 3.2764833
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1887763, upper bound: 3.2801345
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7393270, 1.0187854, -0.1627242, 0.1067304, -0.8460573, 1.1815096
1: -1.1469342, 1.4806118, -0.2170347, 0.1627807, -1.3097148, 1.6976465
2: -0.7548296, 1.6063006, -0.1964690, 0.1719500, -0.9267797, 1.8027695
3: -1.9658682, 1.9879127, -0.2843813, 0.1918265, -2.1576941, 2.2722940
4: -1.1782980, 2.1738334, -0.1784653, 0.2082254, -1.3865234, 2.3522987

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888695, upper bound: 3.2768009
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1888695, upper bound: 3.2814532
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2052934, 0.2239504, -0.1807272, 0.1935380, -0.3988314, 0.4046776
1: -0.2904751, 0.3086712, -0.2565311, 0.2655572, -0.5560324, 0.5652023
2: -0.2511725, 0.3277092, -0.2225670, 0.2781789, -0.5293515, 0.5502762
3: -0.3762597, 0.3788806, -0.3289061, 0.3253363, -0.7015960, 0.7077867
4: -0.2353001, 0.4013989, -0.2148219, 0.3406244, -0.5759245, 0.6162208

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3705657
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3707115
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3098595, 0.4358088, -0.1809336, 0.1953560, -0.5052155, 0.6167424
1: -0.5251927, 0.6571004, -0.2576264, 0.2683870, -0.7935797, 0.9147268
2: -0.3664192, 0.6346077, -0.2231489, 0.2815731, -0.6479923, 0.8577566
3: -0.7393374, 0.8656781, -0.3307490, 0.3294386, -1.0687761, 1.1964271
4: -0.4978842, 0.8078210, -0.2160478, 0.3454387, -0.8433229, 1.0238688

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3705657
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3707115
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2052934, 0.2239504, -0.2781630, 0.3767653, -0.5820588, 0.5021133
1: -0.2904751, 0.3086712, -0.4493879, 0.5516261, -0.8421013, 0.7580591
2: -0.2511725, 0.3277092, -0.3250760, 0.5262765, -0.7774490, 0.6527852
3: -0.3762597, 0.3788806, -0.6205348, 0.7273079, -1.1035676, 0.9994154
4: -0.2353001, 0.4013989, -0.4370639, 0.6687642, -0.9040643, 0.8384628

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3208414
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3443260
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3098595, 0.4358088, -0.2756799, 0.3742723, -0.6841317, 0.7114888
1: -0.5251927, 0.6571004, -0.4457710, 0.5493720, -1.0745647, 1.1028714
2: -0.3664192, 0.6346077, -0.3230441, 0.5223691, -0.8887883, 0.9576519
3: -0.7393374, 0.8656781, -0.6147050, 0.7237957, -1.4631331, 1.4803832
4: -0.4978842, 0.8078210, -0.4339948, 0.6633903, -1.1612744, 1.2418158

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3208414
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3443260
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5386268, 0.7488387, -0.1807272, 0.1935380, -0.7321648, 0.9295659
1: -0.8621221, 1.0861620, -0.2565311, 0.2655572, -1.1276792, 1.3426931
2: -0.5715564, 1.1701052, -0.2225670, 0.2781789, -0.8497353, 1.3926723
3: -1.4353672, 1.4464886, -0.3289061, 0.3253363, -1.7607036, 1.7753947
4: -0.8513992, 1.5654299, -0.2148219, 0.3406244, -1.1920236, 1.7802519

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3240554, upper bound: 3.3681560
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3240554, upper bound: 3.3681560
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7592570, 1.0459166, -0.1809336, 0.1953560, -0.9546130, 1.2268502
1: -1.1903858, 1.5348067, -0.2576264, 0.2683870, -1.4587728, 1.7924330
2: -0.7773191, 1.6696713, -0.2231489, 0.2815731, -1.0588920, 1.8928201
3: -2.0553069, 2.0585208, -0.3307490, 0.3294386, -2.3847456, 2.3892698
4: -1.2212212, 2.2624886, -0.2160478, 0.3454387, -1.5666599, 2.4785364

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3250480, upper bound: 3.3684089
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3250480, upper bound: 3.3684089
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5386268, 0.7488387, -0.2781630, 0.3767653, -0.9153922, 1.0270016
1: -0.8621221, 1.0861620, -0.4493879, 0.5516261, -1.4137483, 1.5355499
2: -0.5715564, 1.1701052, -0.3250760, 0.5262765, -1.0978328, 1.4951812
3: -1.4353672, 1.4464886, -0.6205348, 0.7273079, -2.1626749, 2.0670235
4: -0.8513992, 1.5654299, -0.4370639, 0.6687642, -1.5201635, 2.0024939

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2724984, upper bound: 3.3448270
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2724984, upper bound: 3.3465769
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7592570, 1.0459166, -0.2756799, 0.3742723, -1.1335292, 1.3215965
1: -1.1903858, 1.5348067, -0.4457710, 0.5493720, -1.7397578, 1.9805777
2: -0.7773191, 1.6696713, -0.3230441, 0.5223691, -1.2996881, 1.9927154
3: -2.0553069, 2.0585208, -0.6147050, 0.7237957, -2.7791023, 2.6732259
4: -1.2212212, 2.2624886, -0.4339948, 0.6633903, -1.8846115, 2.6964834

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2734862, upper bound: 3.3480141
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2734862, upper bound: 3.3555851
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2781634, 0.3855372, -0.3256121, 0.4523441, -0.7305075, 0.7111492
1: -0.4504371, 0.5509228, -0.5273179, 0.6454527, -1.0958898, 1.0782406
2: -0.3293265, 0.5766950, -0.3703216, 0.6831124, -1.0124389, 0.9470164
3: -0.6637313, 0.7137094, -0.8054354, 0.8430378, -1.5067692, 1.5191448
4: -0.4132532, 0.7322047, -0.4921194, 0.8815075, -1.2947607, 1.2243241

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4946982, 0.6896759, -0.3248676, 0.4519518, -0.9466499, 1.0145435
1: -0.7894505, 1.0001664, -0.5261787, 0.6474106, -1.4368610, 1.5263450
2: -0.5262433, 1.0673066, -0.3700280, 0.6803505, -1.2065938, 1.4373347
3: -1.2908373, 1.3249892, -0.8004783, 0.8444690, -2.1353064, 2.1254675
4: -0.7720682, 1.4152042, -0.4913234, 0.8767196, -1.6487877, 1.9065276

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2781634, 0.3855372, -0.2484223, 0.3421082, -0.6202716, 0.6339594
1: -0.4504371, 0.5509228, -0.3957042, 0.4897000, -0.9401371, 0.9466269
2: -0.3293265, 0.5766950, -0.3046245, 0.5014510, -0.8307775, 0.8813194
3: -0.6637313, 0.7137094, -0.5605547, 0.6275575, -1.2912889, 1.2742641
4: -0.4132532, 0.7322047, -0.3577088, 0.6274402, -1.0406934, 1.0899135

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3465288, upper bound: 3.3770015
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3402902, upper bound: 3.3729782
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4946982, 0.6896759, -0.2524422, 0.3510644, -0.8457626, 0.9421180
1: -0.7894505, 1.0001664, -0.4049446, 0.5039970, -1.2934475, 1.4051110
2: -0.5262433, 1.0673066, -0.3084072, 0.5135351, -1.0397784, 1.3757137
3: -1.2908373, 1.3249892, -0.5747467, 0.6460370, -1.9368743, 1.8997358
4: -0.7720682, 1.4152042, -0.3675345, 0.6434966, -1.4155648, 1.7827386

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3462398, upper bound: 3.3736738
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3400544, upper bound: 3.3709965
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1908454, 0.1384959, -0.3456418, 0.4810718, -0.6719173, 0.4841377
1: -0.2556005, 0.2031493, -0.5610024, 0.6868975, -0.9424980, 0.7641518
2: -0.2296363, 0.2027422, -0.3901013, 0.7285742, -0.9582104, 0.5928435
3: -0.3280645, 0.2440690, -0.8672440, 0.8995361, -1.2276006, 1.1113130
4: -0.2086726, 0.2360722, -0.5257008, 0.9459569, -1.1546295, 0.7617730

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3539042, upper bound: 3.3416274
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3539877, upper bound: 3.3417495
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2053175, 0.2180639, -0.2697954, 0.3749946, -0.5803121, 0.4878592
1: -0.2863048, 0.2919560, -0.4367954, 0.5320600, -0.8183649, 0.7287514
2: -0.2502842, 0.3177979, -0.3219306, 0.5583752, -0.8086594, 0.6397285
3: -0.3700039, 0.3552795, -0.6400712, 0.6883482, -1.0583521, 0.9953507
4: -0.2326328, 0.3845142, -0.3984006, 0.7076451, -0.9402779, 0.7829148

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3507268, upper bound: 3.3349340
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3512544, upper bound: 3.3349740
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1908454, 0.1384959, -0.2695478, 0.3760729, -0.5669183, 0.4080437
1: -0.2556005, 0.2031493, -0.4326958, 0.5375808, -0.7931814, 0.6358452
2: -0.2296363, 0.2027422, -0.3200988, 0.5548421, -0.7844784, 0.5228409
3: -0.3280645, 0.2440690, -0.6285393, 0.6923354, -1.0203998, 0.8726083
4: -0.2086726, 0.2360722, -0.3963178, 0.7018194, -0.9104920, 0.6323900

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2053175, 0.2180639, -0.2092264, 0.2504458, -0.4557634, 0.4272903
1: -0.2863048, 0.2919560, -0.3016140, 0.3417601, -0.6280649, 0.5935700
2: -0.2502842, 0.3177979, -0.2575050, 0.3613721, -0.6116562, 0.5753029
3: -0.3700039, 0.3552795, -0.3914973, 0.4247779, -0.7947817, 0.7467768
4: -0.2326328, 0.3845142, -0.2490113, 0.4423591, -0.6749918, 0.6335256

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3199837, upper bound: 3.3139758
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120290, upper bound: 3.3120219
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2781634, 0.3855372, -0.3547383, 0.4877394, -0.7659028, 0.7402754
1: -0.4504371, 0.5509228, -0.5790566, 0.6593912, -1.1098282, 1.1299794
2: -0.3293265, 0.5766950, -0.4016187, 0.7437387, -1.0730652, 0.9783136
3: -0.6637313, 0.7137094, -0.9179308, 0.8781205, -1.5418519, 1.6316398
4: -0.4132532, 0.7322047, -0.5479854, 0.9929651, -1.4062183, 1.2801901

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4946982, 0.6896759, -0.3427300, 0.4704646, -0.9651628, 1.0324055
1: -0.7894505, 1.0001664, -0.5584021, 0.6396714, -1.4291220, 1.5585685
2: -0.5262433, 1.0673066, -0.3904814, 0.7146818, -1.2409251, 1.4577881
3: -1.2908373, 1.3249892, -0.8769169, 0.8490540, -2.1398914, 2.2019060
4: -0.7720682, 1.4152042, -0.5283740, 0.9490764, -1.7211442, 1.9435782

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3218176, upper bound: 3.3488809
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3218176, upper bound: 3.3548954
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1978077, 0.1756163, -0.3664802, 0.5043814, -0.7021891, 0.5420965
1: -0.2694665, 0.2450498, -0.6070046, 0.6854192, -0.9548857, 0.8520544
2: -0.2390177, 0.2552907, -0.4137102, 0.7665520, -1.0055697, 0.6690008
3: -0.3444529, 0.2917261, -0.9603794, 0.9151838, -1.2596366, 1.2521055
4: -0.2195973, 0.3011988, -0.5638500, 1.0202947, -1.2398920, 0.8650486

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2463130, 0.3180011, -0.3664802, 0.5043814, -0.7506944, 0.6844813
1: -0.3736300, 0.4293446, -0.6070046, 0.6854192, -1.0590491, 1.0363491
2: -0.3014385, 0.4796235, -0.4137102, 0.7665520, -1.0679905, 0.8933337
3: -0.5487667, 0.5477364, -0.9603794, 0.9151838, -1.4639505, 1.5081155
4: -0.3184631, 0.5994291, -0.5638500, 1.0202947, -1.3387578, 1.1632791

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3494082
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3502041
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1908454, 0.1384959, -0.4467238, 0.6215951, -0.8124405, 0.5852197
1: -0.2556005, 0.2031493, -0.7328915, 0.8427354, -1.0983360, 0.9360408
2: -0.2296363, 0.2027422, -0.4924225, 0.9656458, -1.1952821, 0.6951647
3: -0.3280645, 0.2440690, -1.2085794, 1.1289421, -1.4570067, 1.4526484
4: -0.2086726, 0.2360722, -0.7015048, 1.3040695, -1.5127420, 0.9375770

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2053175, 0.2180639, -0.4467238, 0.6215951, -0.8269126, 0.6647877
1: -0.2863048, 0.2919560, -0.7328915, 0.8427354, -1.1290402, 1.0248474
2: -0.2502842, 0.3177979, -0.4924225, 0.9656458, -1.2159300, 0.8102203
3: -0.3700039, 0.3552795, -1.2085794, 1.1289421, -1.4989460, 1.5638589
4: -0.2326328, 0.3845142, -0.7015048, 1.3040695, -1.5367023, 1.0860190

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1908454, 0.1384959, -0.4117214, 0.5673239, -0.7581693, 0.5502174
1: -0.2556005, 0.2031493, -0.6853573, 0.7768623, -1.0324628, 0.8885065
2: -0.2296363, 0.2027422, -0.4597958, 0.8666953, -1.0963316, 0.6625379
3: -0.3280645, 0.2440690, -1.0987918, 1.0422142, -1.3702786, 1.3428607
4: -0.2086726, 0.2360722, -0.6412443, 1.1624334, -1.3711059, 0.8773165

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2036928, 0.2124113, -0.4117214, 0.5673239, -0.7710166, 0.6241327
1: -0.2829069, 0.2859761, -0.6853573, 0.7768623, -1.0597692, 0.9713334
2: -0.2480286, 0.3095272, -0.4597958, 0.8666953, -1.1147239, 0.7693230
3: -0.3657826, 0.3466436, -1.0987918, 1.0422142, -1.4079967, 1.4454354
4: -0.2295885, 0.3734564, -0.6412443, 1.1624334, -1.3920219, 1.0147008

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4744946, 0.6584470, -0.3456418, 0.4810718, -0.9555664, 1.0040888
1: -0.7762834, 0.8887688, -0.5610024, 0.6868975, -1.4631809, 1.4497712
2: -0.5175221, 1.0314066, -0.3901013, 0.7285742, -1.2460963, 1.4215080
3: -1.2997166, 1.1929538, -0.8672440, 0.8995361, -2.1992526, 2.0601978
4: -0.7383318, 1.3979359, -0.5257008, 0.9459569, -1.6842887, 1.9236367

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3148019
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3167523
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3331804, 0.4553815, -0.3456418, 0.4810718, -0.8142522, 0.8010234
1: -0.5395908, 0.6057771, -0.5610024, 0.6868975, -1.2264882, 1.1667795
2: -0.3794020, 0.6958874, -0.3901013, 0.7285742, -1.1079761, 1.0859888
3: -0.8536465, 0.8025000, -0.8672440, 0.8995361, -1.7531826, 1.6697439
4: -0.5008957, 0.9233500, -0.5257008, 0.9459569, -1.4468527, 1.4490508

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3167757
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3182551
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4787475, 0.6936553, -0.2695478, 0.3760729, -0.8548204, 0.9632028
1: -0.7830029, 0.9466329, -0.4326958, 0.5375808, -1.3205836, 1.3793287
2: -0.5240572, 1.0692931, -0.3200988, 0.5548421, -1.0788993, 1.3893919
3: -1.3251177, 1.2538862, -0.6285393, 0.6923354, -2.0174532, 1.8824255
4: -0.7577459, 1.4301879, -0.3963178, 0.7018194, -1.4595652, 1.8265058

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3331804, 0.4553815, -0.2695478, 0.3760729, -0.7092532, 0.7249293
1: -0.5395908, 0.6057771, -0.4326958, 0.5375808, -1.0771716, 1.0384730
2: -0.3794020, 0.6958874, -0.3200988, 0.5548421, -0.9342440, 1.0159861
3: -0.8536465, 0.8025000, -0.6285393, 0.6923354, -1.5459819, 1.4310390
4: -0.5008957, 0.9233500, -0.3963178, 0.7018194, -1.2027152, 1.3196677

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5023870, 0.6938046, -0.2697954, 0.3749946, -0.8773816, 0.9636000
1: -0.8422163, 0.9537991, -0.4367954, 0.5320600, -1.3742760, 1.3905945
2: -0.5516966, 1.0765066, -0.3219306, 0.5583752, -1.1100717, 1.3984373
3: -1.3880401, 1.2902215, -0.6400712, 0.6883482, -2.0763876, 1.9302926
4: -0.7910100, 1.4597491, -0.3984006, 0.7076451, -1.4986551, 1.8581496

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3205358, 0.4363322, -0.2697954, 0.3749946, -0.6955304, 0.7061275
1: -0.5238961, 0.5892826, -0.4367954, 0.5320600, -1.0559561, 1.0260780
2: -0.3660394, 0.6587486, -0.3219306, 0.5583752, -0.9244146, 0.9806793
3: -0.8100439, 0.7791889, -0.6400712, 0.6883482, -1.4983921, 1.4192600
4: -0.4773532, 0.8607262, -0.3984006, 0.7076451, -1.1849983, 1.2591268

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5255073, 0.7187296, -0.2092264, 0.2504458, -0.7759531, 0.9279560
1: -0.8657568, 0.9886846, -0.3016140, 0.3417601, -1.2075169, 1.2902986
2: -0.5684106, 1.1250722, -0.2575050, 0.3613721, -0.9297827, 1.3825772
3: -1.4418929, 1.3343377, -0.3914973, 0.4247779, -1.8666705, 1.7258351
4: -0.8186107, 1.5249867, -0.2490113, 0.4423591, -1.2609698, 1.7739980

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3205358, 0.4363322, -0.2092264, 0.2504458, -0.5709817, 0.6455586
1: -0.5238961, 0.5892826, -0.3016140, 0.3417601, -0.8656562, 0.8908966
2: -0.3660394, 0.6587486, -0.2575050, 0.3613721, -0.7274114, 0.9162536
3: -0.8100439, 0.7791889, -0.3914973, 0.4247779, -1.2348218, 1.1706862
4: -0.4773532, 0.8607262, -0.2490113, 0.4423591, -0.9197123, 1.1097375

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4787475, 0.6936553, -0.3730532, 0.5142136, -0.9929610, 1.0667084
1: -0.7830029, 0.9466329, -0.6110075, 0.6962433, -1.4792463, 1.5576403
2: -0.5240572, 1.0692931, -0.4200994, 0.7856184, -1.3096755, 1.4893925
3: -1.3251177, 1.2538862, -0.9770764, 0.9289197, -2.2540374, 2.2309623
4: -0.7577459, 1.4301879, -0.5794039, 1.0543439, -1.8120897, 2.0095918

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3132820
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3132820
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3331804, 0.4553815, -0.4467238, 0.6215951, -0.9547754, 0.9021053
1: -0.5395908, 0.6057771, -0.7328915, 0.8427354, -1.3823261, 1.3386682
2: -0.3794020, 0.6958874, -0.4924225, 0.9656458, -1.3450478, 1.1883099
3: -0.8536465, 0.8025000, -1.2085794, 1.1289421, -1.9825885, 2.0110793
4: -0.5008957, 0.9233500, -0.7015048, 1.3040695, -1.8049651, 1.6248547

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3133621
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3133706
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4787475, 0.6936553, -0.3664802, 0.5043814, -0.9831288, 1.0601356
1: -0.7830029, 0.9466329, -0.6070046, 0.6854192, -1.4684218, 1.5536375
2: -0.5240572, 1.0692931, -0.4137102, 0.7665520, -1.2906091, 1.4830031
3: -1.3251177, 1.2538862, -0.9603794, 0.9151838, -2.2403016, 2.2142656
4: -0.7577459, 1.4301879, -0.5638500, 1.0202947, -1.7780402, 1.9940379

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3130727
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3130727
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3331804, 0.4553815, -0.4117214, 0.5673239, -0.9005042, 0.8671029
1: -0.5395908, 0.6057771, -0.6853573, 0.7768623, -1.3164530, 1.2911342
2: -0.3794020, 0.6958874, -0.4597958, 0.8666953, -1.2460973, 1.1556833
3: -0.8536465, 0.8025000, -1.0987918, 1.0422142, -1.8958607, 1.9012918
4: -0.5008957, 0.9233500, -0.6412443, 1.1624334, -1.6633291, 1.5645943

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5023870, 0.6938046, -0.6279464, 0.9307750, -1.4331620, 1.3217511
1: -0.8422163, 0.9537991, -1.0247976, 1.3076565, -2.1498728, 1.9785964
2: -0.5516966, 1.0765066, -0.6732014, 1.4444041, -1.9961007, 1.7497078
3: -1.3880401, 1.2902215, -1.7774800, 1.7267673, -3.1148062, 3.0677011
4: -0.7910100, 1.4597491, -1.0197093, 1.9138947, -2.7049043, 2.4794583

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3205358, 0.4363322, -0.6279464, 0.9307750, -1.2513108, 1.0642786
1: -0.5238961, 0.5892826, -1.0247976, 1.3076565, -1.8315526, 1.6140802
2: -0.3660394, 0.6587486, -0.6732014, 1.4444041, -1.8104435, 1.3319499
3: -0.8100439, 0.7791889, -1.7774800, 1.7267673, -2.5368111, 2.5566685
4: -0.4773532, 0.8607262, -1.0197093, 1.9138947, -2.3912477, 1.8804355

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5255073, 0.7187296, -0.3747006, 0.5153155, -1.0408227, 1.0934303
1: -0.8657568, 0.9886846, -0.6229787, 0.7066517, -1.5724084, 1.6116633
2: -0.5684106, 1.1250722, -0.4230216, 0.7808440, -1.3492546, 1.5480938
3: -1.4418929, 1.3343377, -0.9837797, 0.9448279, -2.3867199, 2.3181174
4: -0.8186107, 1.5249867, -0.5836946, 1.0469630, -1.8655736, 2.1086812

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3205358, 0.4363322, -0.3747006, 0.5153155, -0.8358514, 0.8110327
1: -0.5238961, 0.5892826, -0.6229787, 0.7066517, -1.2305479, 1.2122614
2: -0.3660394, 0.6587486, -0.4230216, 0.7808440, -1.1468835, 1.0817702
3: -0.8100439, 0.7791889, -0.9837797, 0.9448279, -1.7548717, 1.7629683
4: -0.4773532, 0.8607262, -0.5836946, 1.0469630, -1.5243162, 1.4444205

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.35 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.04 seconds
NS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
NS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1357337
NS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
NS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.1366150
NS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2329620
NS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2329620
NS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2581693
NS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0607899, upper bound: 3.2581693
NS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1357337, upper bound: 3.0607899
NS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1357337, upper bound: 3.0610602
NS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2329620, upper bound: 3.1498572
NS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2329620, upper bound: 3.1864269
NS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0162219, upper bound: 3.0162219
NS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0162219, upper bound: 3.0463998
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0463998, upper bound: 3.0939294
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.0463998, upper bound: 3.1566236
NS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2953006, upper bound: 3.2812533
NS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2813166
NS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2221127, upper bound: 3.2601534
NS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2489007, upper bound: 3.2678566
NS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2586818, upper bound: 3.1847512
NS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1849856
NS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.04
Output dim: 0, lower bound: -2.8137068, upper bound: 2.8203866
NS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2224935, upper bound: 3.1769383
NS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3416311, upper bound: 3.3037967
NS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3258777, upper bound: 3.3007828
NS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3390803, upper bound: 3.2884686
NS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3232250, upper bound: 3.2844933
NS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2875735
NS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2909828
NS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2875735
NS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3382464, upper bound: 3.2909828
NS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3178212
NS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3705657, upper bound: 3.3178212
NS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3178212
NS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3707115, upper bound: 3.3178212
NS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3707288, upper bound: 3.3260525
NS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3707288, upper bound: 3.3260525
NS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3709153, upper bound: 3.3260525
NS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3709153, upper bound: 3.3260525
NS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208409, upper bound: 3.2386825
NS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208409, upper bound: 3.2386825
NS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2667848
NS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3443260, upper bound: 3.2683777
NS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3516686, upper bound: 3.2744858
NS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3516686, upper bound: 3.2744858
NS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3602240, upper bound: 3.3176191
NS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3602240, upper bound: 3.3182276
NS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
NS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000038, upper bound: 3.3662019
NS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
NS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662019
NS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2769040
NS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2818812
NS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2769040
NS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888890, upper bound: 3.2818812
NS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
NS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
NS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
NS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3000040, upper bound: 3.3662746
NS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1887763, upper bound: 3.2764833
NS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1887763, upper bound: 3.2801345
NS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888695, upper bound: 3.2768009
NS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.1888695, upper bound: 3.2814532
NS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3705657
NS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3707115
NS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3705657
NS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3178212, upper bound: 3.3707115
NS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3208414
NS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3443260
NS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3208414
NS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2386829, upper bound: 3.3443260
NS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3240554, upper bound: 3.3681560
NS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3240554, upper bound: 3.3681560
NS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3250480, upper bound: 3.3684089
NS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3250480, upper bound: 3.3684089
NS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2724984, upper bound: 3.3448270
NS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2724984, upper bound: 3.3465769
NS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2734862, upper bound: 3.3480141
NS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.2734862, upper bound: 3.3555851
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3904408, upper bound: 3.3903893
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3465288, upper bound: 3.3770015
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3402902, upper bound: 3.3729782
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3462398, upper bound: 3.3736738
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3400544, upper bound: 3.3709965
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3539042, upper bound: 3.3416274
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3539877, upper bound: 3.3417495
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3507268, upper bound: 3.3349340
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3512544, upper bound: 3.3349740
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3320384, upper bound: 3.3319424
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3199837, upper bound: 3.3139758
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3120290, upper bound: 3.3120219
NS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3218176, upper bound: 3.3488809
NS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3218176, upper bound: 3.3548954
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3329774
NS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3494082
NS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3174855, upper bound: 3.3502041
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
NS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
NS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3150447, upper bound: 3.3209046
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3142847, upper bound: 3.3207963
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3148019
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3167523
NS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3167757
NS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3208035, upper bound: 3.3182551
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3146436
NS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
NS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3200962, upper bound: 3.3151464
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3189383, upper bound: 3.3136111
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3132820
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3132820
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3133621
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3132820, upper bound: 3.3133706
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3130727
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3130727
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3124994, upper bound: 3.3131396
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3127542, upper bound: 3.3123985
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.04
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.0637031, 0.0077558, -0.0702763, 0.0653643
1: -0.0692971, 0.0076117, -0.0709020, 0.0146530, -0.0839502, 0.0785138
2: -0.0580488, 0.0014766, -0.0592750, 0.0115783, -0.0696271, 0.0607516
3: -0.0626092, 0.0125836, -0.0655022, 0.0203997, -0.0830089, 0.0780858
4: -0.0435550, 0.0022268, -0.0454009, 0.0171351, -0.0606901, 0.0476277

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9755296, upper bound: 3.0518208
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9752504, upper bound: 3.0512659
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.1370131, 0.0429107, -0.1054312, 0.1386743
1: -0.0692971, 0.0076117, -0.1863651, 0.0698088, -0.1391059, 0.1939768
2: -0.0580488, 0.0014766, -0.1751275, 0.0662455, -0.1242944, 0.1766041
3: -0.0626092, 0.0125836, -0.1781025, 0.0912109, -0.1538201, 0.1906861
4: -0.0435550, 0.0022268, -0.1064398, 0.0848362, -0.1283912, 0.1086666

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9755296, upper bound: 3.0518208
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9752504, upper bound: 3.0512659
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0664968, 0.0116788, -0.0637031, 0.0077558, -0.0742527, 0.0753818
1: -0.0752405, 0.0252749, -0.0709020, 0.0146530, -0.0898935, 0.0961770
2: -0.0628744, 0.0152150, -0.0592750, 0.0115783, -0.0744527, 0.0744900
3: -0.0685259, 0.0332320, -0.0655022, 0.0203997, -0.0889256, 0.0987342
4: -0.0497595, 0.0198232, -0.0454009, 0.0171351, -0.0668946, 0.0652241

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9755685, upper bound: 3.0522602
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9735033, upper bound: 3.0466512
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0664968, 0.0116788, -0.1370131, 0.0429107, -0.1094075, 0.1486919
1: -0.0752405, 0.0252749, -0.1863651, 0.0698088, -0.1450493, 0.2116400
2: -0.0628744, 0.0152150, -0.1751275, 0.0662455, -0.1291199, 0.1903426
3: -0.0685259, 0.0332320, -0.1781025, 0.0912109, -0.1597368, 0.2113345
4: -0.0497595, 0.0198232, -0.1064398, 0.0848362, -0.1345958, 0.1262630

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9755685, upper bound: 3.0522602
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9735033, upper bound: 3.0466512
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.0666025, 0.0125595, -0.0750800, 0.0682636
1: -0.0692971, 0.0076117, -0.0755262, 0.0264283, -0.0957255, 0.0831379
2: -0.0580488, 0.0014766, -0.0630669, 0.0163602, -0.0744090, 0.0645435
3: -0.0626092, 0.0125836, -0.0687606, 0.0340288, -0.0966380, 0.0813443
4: -0.0435550, 0.0022268, -0.0499745, 0.0211048, -0.0646598, 0.0522013

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9834151, upper bound: 3.0593684
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835629, upper bound: 3.0595902
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0625205, 0.0016612, -0.1596637, 0.0941035, -0.1566240, 0.1613248
1: -0.0692971, 0.0076117, -0.2106493, 0.1473507, -0.2166478, 0.2182610
2: -0.0580488, 0.0014766, -0.1921865, 0.1520855, -0.2101343, 0.1936630
3: -0.0626092, 0.0125836, -0.2745035, 0.1743023, -0.2369114, 0.2870872
4: -0.0435550, 0.0022268, -0.1750863, 0.1810839, -0.2246389, 0.1773131

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9834151, upper bound: 3.0593684
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835629, upper bound: 3.0595902
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0666025, 0.0125595, -0.0666025, 0.0125595, -0.0791619, 0.0791619
1: -0.0755262, 0.0264283, -0.0755262, 0.0264283, -0.1019545, 0.1019545
2: -0.0630669, 0.0163602, -0.0630669, 0.0163602, -0.0794271, 0.0794271
3: -0.0687606, 0.0340288, -0.0687606, 0.0340288, -0.1027895, 0.1027895
4: -0.0499745, 0.0211048, -0.0499745, 0.0211048, -0.0710793, 0.0710793

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1818393, upper bound: 3.2484431
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1825668, upper bound: 3.2471914
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0666025, 0.0125595, -0.1596637, 0.0941035, -0.1607059, 0.1722231
1: -0.0755262, 0.0264283, -0.2106493, 0.1473507, -0.2228768, 0.2370776
2: -0.0630669, 0.0163602, -0.1921865, 0.1520855, -0.2151524, 0.2085466
3: -0.0687606, 0.0340288, -0.2745035, 0.1743023, -0.2430629, 0.3085324
4: -0.0499745, 0.0211048, -0.1750863, 0.1810839, -0.2310584, 0.1961911

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1818393, upper bound: 3.2484431
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1825668, upper bound: 3.2471914
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.0625205, 0.0016612, -0.1130076, 0.0992431
1: -0.1613708, 0.0607112, -0.0692971, 0.0076117, -0.1689826, 0.1300083
2: -0.1328885, 0.0555079, -0.0580488, 0.0014766, -0.1343651, 0.1135567
3: -0.1744612, 0.0704553, -0.0626092, 0.0125836, -0.1870448, 0.1330645
4: -0.0577233, 0.0723308, -0.0435550, 0.0022268, -0.0599501, 0.1158858

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0784281, upper bound: 3.0114707
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0466512, upper bound: 2.9735033
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1113464, 0.0367225, -0.0664968, 0.0116788, -0.1230251, 0.1032194
1: -0.1613708, 0.0607112, -0.0752405, 0.0252749, -0.1866458, 0.1359516
2: -0.1328885, 0.0555079, -0.0628744, 0.0152150, -0.1481035, 0.1183822
3: -0.1744612, 0.0704553, -0.0685259, 0.0332320, -0.2076932, 0.1389812
4: -0.0577233, 0.0723308, -0.0497595, 0.0198232, -0.0775465, 0.1220903

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0784281, upper bound: 3.0114707
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0466512, upper bound: 2.9735033
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.0625205, 0.0016612, -0.1613248, 0.1566240
1: -0.2106493, 0.1473507, -0.0692971, 0.0076117, -0.2182610, 0.2166478
2: -0.1921865, 0.1520855, -0.0580488, 0.0014766, -0.1936630, 0.2101343
3: -0.2745035, 0.1743023, -0.0626092, 0.0125836, -0.2870872, 0.2369114
4: -0.1750863, 0.1810839, -0.0435550, 0.0022268, -0.1773131, 0.2246389

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1620772, upper bound: 3.0825263
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0595902, upper bound: 2.9835629
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.0666025, 0.0125595, -0.1722231, 0.1607059
1: -0.2106493, 0.1473507, -0.0755262, 0.0264283, -0.2370776, 0.2228768
2: -0.1921865, 0.1520855, -0.0630669, 0.0163602, -0.2085466, 0.2151524
3: -0.2745035, 0.1743023, -0.0687606, 0.0340288, -0.3085324, 0.2430629
4: -0.1750863, 0.1810839, -0.0499745, 0.0211048, -0.1961911, 0.2310584

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1620772, upper bound: 3.0904380
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0595902, upper bound: 2.9835629
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.1113464, 0.0367225, -0.1963862, 0.2054498
1: -0.2106493, 0.1473507, -0.1613708, 0.0607112, -0.2713604, 0.3087215
2: -0.1921865, 0.1520855, -0.1328885, 0.0555079, -0.2476943, 0.2849740
3: -0.2745035, 0.1743023, -0.1744612, 0.0704553, -0.3449588, 0.3487634
4: -0.1750863, 0.1810839, -0.0577233, 0.0723308, -0.2474171, 0.2388072

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9553184, upper bound: 3.0018614
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9268942, upper bound: 2.9303024
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.1596637, 0.0941035, -0.2537671, 0.2537671
1: -0.2106493, 0.1473507, -0.2106493, 0.1473507, -0.3579999, 0.3579999
2: -0.1921865, 0.1520855, -0.1921865, 0.1520855, -0.3442720, 0.3442720
3: -0.2745035, 0.1743023, -0.2745035, 0.1743023, -0.4488058, 0.4488058
4: -0.1750863, 0.1810839, -0.1750863, 0.1810839, -0.3561702, 0.3561702

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9553184, upper bound: 3.0090852
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9268942, upper bound: 2.9328373
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0661732, 0.0095017, -0.1712859, 0.1554839, -0.2216571, 0.1807876
1: -0.0747617, 0.0214105, -0.2360464, 0.2119974, -0.2867591, 0.2574569
2: -0.0624904, 0.0124668, -0.2078791, 0.2236603, -0.2861507, 0.2203458
3: -0.0678525, 0.0287675, -0.3012584, 0.2526170, -0.3204695, 0.3300259
4: -0.0486811, 0.0167277, -0.1917115, 0.2698154, -0.3184964, 0.2084392

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2812533
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2812533
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0661784, 0.0100689, -0.2038606, 0.2919195, -0.3580979, 0.2139295
1: -0.0747685, 0.0223205, -0.3115785, 0.3982663, -0.4730348, 0.3338990
2: -0.0624953, 0.0131260, -0.2587352, 0.4161468, -0.4786421, 0.2718611
3: -0.0678984, 0.0296135, -0.4173448, 0.5144377, -0.5823361, 0.4469583
4: -0.0488297, 0.0172780, -0.2883344, 0.5225226, -0.5713522, 0.3056125

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2813166
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2944464, upper bound: 3.2813166
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0661732, 0.0095017, -0.2390835, 0.3211969, -0.3873701, 0.2485852
1: -0.0747617, 0.0214105, -0.3812199, 0.4675027, -0.5422643, 0.4026304
2: -0.0624904, 0.0124668, -0.2864437, 0.4499321, -0.5124225, 0.2989104
3: -0.0678525, 0.0287675, -0.5193738, 0.6089930, -0.6768456, 0.5481413
4: -0.0486811, 0.0167277, -0.3643858, 0.5651127, -0.6137937, 0.3811135

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2221127, upper bound: 3.2601534
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2221127, upper bound: 3.2601534
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0661784, 0.0100689, -0.2935781, 0.4094446, -0.4756230, 0.3036470
1: -0.0747685, 0.0223205, -0.4827732, 0.5984747, -0.6732432, 0.5050938
2: -0.0624953, 0.0131260, -0.3439355, 0.5752233, -0.6377186, 0.3570615
3: -0.0678984, 0.0296135, -0.6737927, 0.7971052, -0.8650036, 0.7034062
4: -0.0488297, 0.0172780, -0.4793114, 0.7390931, -0.7879227, 0.4965895

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2476269, upper bound: 3.2678566
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2476269, upper bound: 3.2678566
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1578780, 0.0850560, -0.1712859, 0.1554839, -0.3133619, 0.2563419
1: -0.2068545, 0.1356614, -0.2360464, 0.2119974, -0.4188519, 0.3717077
2: -0.1900446, 0.1382954, -0.2078791, 0.2236603, -0.4137049, 0.3461744
3: -0.2682851, 0.1606151, -0.3012584, 0.2526170, -0.5209020, 0.4618735
4: -0.1726831, 0.1627396, -0.1917115, 0.2698154, -0.4424985, 0.3544511

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1834310
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1847512
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1579494, 0.0841533, -0.2038606, 0.2919195, -0.4498689, 0.2880139
1: -0.2069035, 0.1348604, -0.3115785, 0.3982663, -0.6051698, 0.4464388
2: -0.1901306, 0.1368759, -0.2587352, 0.4161468, -0.6062775, 0.3956110
3: -0.2681317, 0.1595277, -0.4173448, 0.5144377, -0.7825694, 0.5768725
4: -0.1726355, 0.1609216, -0.2883344, 0.5225226, -0.6951580, 0.4492561

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1835651
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580489, upper bound: 3.1849856
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1596637, 0.0941035, -0.2763702, 0.3737459, -0.5334095, 0.3704736
1: -0.2106493, 0.1473507, -0.4448302, 0.5465982, -0.7572474, 0.5921809
2: -0.1921865, 0.1520855, -0.3230560, 0.5219079, -0.7140944, 0.4751415
3: -0.2745035, 0.1743023, -0.6137326, 0.7200914, -0.9945949, 0.7880348
4: -0.1750863, 0.1810839, -0.4328698, 0.6627753, -0.8378615, 0.6139537

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2180576, upper bound: 3.1725192
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2211870, upper bound: 3.1755080
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1454872, 0.0494159, -0.4160951, 0.5804975, -0.7259847, 0.4655110
1: -0.1985075, 0.0852132, -0.6700065, 0.7745265, -0.9730340, 0.7552198
2: -0.1861054, 0.0731980, -0.4566320, 0.9105017, -1.0966070, 0.5298299
3: -0.1893635, 0.1107613, -1.1147693, 1.0319089, -1.2212723, 1.2255306
4: -0.1150604, 0.0980075, -0.6346505, 1.2213130, -1.3363733, 0.7326580

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1479754, 0.0622823, -0.2819788, 0.3823992, -0.5303746, 0.3442611
1: -0.2030102, 0.1032830, -0.4470777, 0.5044666, -0.7074768, 0.5503607
2: -0.1896725, 0.0943340, -0.3291067, 0.5854331, -0.7751055, 0.4234408
3: -0.1976196, 0.1311589, -0.6892278, 0.6617533, -0.8593729, 0.8203866
4: -0.1192789, 0.1223733, -0.4078562, 0.7639912, -0.8832701, 0.5302294

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1503968, 0.1109050, -0.3995670, 0.5569234, -0.7073202, 0.5104719
1: -0.2125855, 0.1644968, -0.6426897, 0.7464386, -0.9590241, 0.8071864
2: -0.1936026, 0.1698938, -0.4415951, 0.8701487, -1.0637513, 0.6114889
3: -0.2214947, 0.2031720, -1.0599607, 0.9921119, -1.2136067, 1.2631325
4: -0.1321175, 0.2029141, -0.6088501, 1.1627142, -1.2948318, 0.8117642

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1530817, 0.1312132, -0.2737684, 0.3690525, -0.5221342, 0.4049816
1: -0.2193033, 0.1904956, -0.4316627, 0.4891993, -0.7085026, 0.6221583
2: -0.1977287, 0.2014404, -0.3233541, 0.5630678, -0.7607964, 0.5247945
3: -0.2352768, 0.2330002, -0.6586707, 0.6397146, -0.8749914, 0.8916709
4: -0.1397993, 0.2402222, -0.3932930, 0.7316470, -0.8714464, 0.6335152

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1479754, 0.0622823, -0.6467208, 0.9910436, -1.1390190, 0.7090030
1: -0.2030102, 0.1032830, -1.0478625, 1.4110174, -1.6140276, 1.1511453
2: -0.1896725, 0.0943340, -0.6901353, 1.5304074, -1.7200799, 0.7844693
3: -0.1976196, 0.1311589, -1.8405366, 1.8416318, -2.0392513, 1.9716953
4: -0.1192789, 0.1223733, -1.0482036, 2.0166445, -2.1359234, 1.1705766

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1479754, 0.0622823, -0.8430368, 1.3377683, -1.4857438, 0.9053190
1: -0.2030102, 0.1032830, -1.3349321, 1.9141948, -2.1172051, 1.4382150
2: -0.1896725, 0.0943340, -0.8788161, 2.0659001, -2.2555726, 0.9731500
3: -0.1976196, 0.1311589, -2.4155483, 2.4862275, -2.6838472, 2.5467072
4: -0.1192789, 0.1223733, -1.3854833, 2.6965194, -2.8157983, 1.5078565

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1530817, 0.1312132, -0.6467208, 0.9910436, -1.1441252, 0.7779340
1: -0.2193033, 0.1904956, -1.0478625, 1.4110174, -1.6303207, 1.2383580
2: -0.1977287, 0.2014404, -0.6901353, 1.5304074, -1.7281361, 0.8915757
3: -0.2352768, 0.2330002, -1.8405366, 1.8416318, -2.0769086, 2.0735369
4: -0.1397993, 0.2402222, -1.0482036, 2.0166445, -2.1564438, 1.2884258

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1530817, 0.1312132, -0.8430368, 1.3377683, -1.4908500, 0.9742500
1: -0.2193033, 0.1904956, -1.3349321, 1.9141948, -2.1334982, 1.5254277
2: -0.1977287, 0.2014404, -0.8788161, 2.0659001, -2.2636287, 1.0802565
3: -0.2352768, 0.2330002, -2.4155483, 2.4862275, -2.7215044, 2.6485486
4: -0.1397993, 0.2402222, -1.3854833, 2.6965194, -2.8363187, 1.6257055

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.85 + 418.37 = 420.22 seconds
