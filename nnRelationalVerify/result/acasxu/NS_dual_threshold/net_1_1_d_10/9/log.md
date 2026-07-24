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
execution time: IAR + RelationalAnalysis = 0.75 + 1.17 = 1.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982239, upper bound: 3.3982239

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3971201, upper bound: 3.3971414
time: 0.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3967501, upper bound: 3.3967501
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -3.3971201, upper bound: 3.3971414
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -3.3967501, upper bound: 3.3967501

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.1802469, 2.1300411, -1.2466472, 2.2771921, -3.4574389, 3.3766885
1: -1.8617817, 2.9912033, -1.9637374, 3.1950235, -5.0568037, 4.9549408
2: -1.2867744, 3.0743372, -1.3604455, 3.2710245, -4.5577965, 4.4347830
3: -3.2712717, 3.8114285, -3.4595599, 4.0564485, -7.3277192, 7.2709880
4: -2.0556865, 3.9620397, -2.1785955, 4.2066536, -6.2623401, 6.1406355

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3221417, upper bound: 3.3727322
time: 0.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3903298, upper bound: 3.3914012
time: 0.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.6433699, 3.1101141, -1.1639766, 2.0932665, -3.7366364, 4.2740908
1: -2.5895941, 4.3057761, -1.8360945, 2.9443133, -5.5339069, 6.1418705
2: -1.8265626, 4.4649234, -1.2657543, 3.0278146, -4.8543773, 5.7306776
3: -4.5898609, 5.4298372, -3.2263303, 3.7549329, -8.3447933, 8.6561680
4: -2.9406917, 5.6675196, -2.0244493, 3.9050744, -6.8457661, 7.6919689

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0038287, upper bound: 3.0849111
time: 0.22 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9500734, upper bound: 2.9500734
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -3.3221417, upper bound: 3.3727322
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -3.3903298, upper bound: 3.3914012
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -3.0038287, upper bound: 3.0849111
NS_A2_A2, status: Status.VERIFIED, split count: 2, time: 1.31
Output dim: 0, lower bound: -2.9500734, upper bound: 2.9500734

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.1802469, 2.1300411, -0.2895984, 0.3943615, -1.5746084, 2.4196393
1: -1.8617817, 2.9912033, -0.4578271, 0.5641531, -2.4259348, 3.4490302
2: -1.2867744, 3.0743372, -0.3370600, 0.5814233, -1.8681974, 3.4113965
3: -3.2712717, 3.8114285, -0.6758766, 0.7399411, -4.0112128, 4.4873052
4: -2.0556865, 3.9620397, -0.4528955, 0.7513722, -2.8070583, 4.4149346

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3059520, upper bound: 3.3102251
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3059520, upper bound: 3.3727322
time: 0.29 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.0127860, 1.7897639, -1.0333478, 1.8749698, -2.8877559, 2.8231118
1: -1.6049254, 2.5288506, -1.6493901, 2.6237774, -4.2287025, 4.1782408
2: -1.0961735, 2.5992632, -1.1160958, 2.7230368, -3.8192103, 3.7153585
3: -2.8085499, 3.2405746, -2.9354391, 3.3341267, -6.1426764, 6.1760120
4: -1.7501594, 3.3732395, -1.7549448, 3.5216486, -5.2718081, 5.1281843

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3728289, upper bound: 3.3281151
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3728289, upper bound: 3.3914012
time: 0.29 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.9253982, 1.6097733, -1.0809948, 1.9203355, -2.8457336, 2.6907678
1: -1.4741561, 2.2789021, -1.7077280, 2.7088583, -4.1830144, 3.9866300
2: -1.0221615, 2.3553617, -1.1711046, 2.7848110, -3.8069723, 3.5264657
3: -2.5333016, 2.9429610, -2.9915843, 3.4651990, -5.9984989, 5.9345455
4: -1.6486474, 3.0149312, -1.8733954, 3.6041093, -5.2527566, 4.8883266

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9437923, upper bound: 3.0217858
time: 0.26 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.31 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -3.3059520, upper bound: 3.3102251
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -3.3059520, upper bound: 3.3727322
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -3.3728289, upper bound: 3.3281151
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -3.3728289, upper bound: 3.3914012
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -2.9437923, upper bound: 3.0217858
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2534180, 0.3392607, -0.2895984, 0.3943615, -0.6477795, 0.6288590
1: -0.3963218, 0.4851879, -0.4578271, 0.5641531, -0.9604748, 0.9430148
2: -0.3020439, 0.4982153, -0.3370600, 0.5814233, -0.8834672, 0.8352751
3: -0.5673350, 0.6312910, -0.6758766, 0.7399411, -1.3072761, 1.3071674
4: -0.3886379, 0.6367542, -0.4528955, 0.7513722, -1.1400101, 1.0896493

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2979264, upper bound: 3.2961051
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3040062
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9670416, 1.7268693, -0.2895984, 0.3943615, -1.3614031, 2.0164676
1: -1.5470181, 2.4186072, -0.4578271, 0.5641531, -2.1111712, 2.8764343
2: -1.0415678, 2.5212839, -0.3370600, 0.5814233, -1.6229911, 2.8583436
3: -2.7453201, 3.0853603, -0.6758766, 0.7399411, -3.4852612, 3.7612369
4: -1.6330600, 3.2696786, -0.4528955, 0.7513722, -2.3844318, 3.7225735

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2908422, upper bound: 3.3611142
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2881803, upper bound: 3.3628144
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2534180, 0.3392607, -1.0333478, 1.8749698, -2.1283879, 1.3726085
1: -0.3963218, 0.4851879, -1.6493901, 2.6237774, -3.0200992, 2.1345775
2: -0.3020439, 0.4982153, -1.1160958, 2.7230368, -3.0250807, 1.6143110
3: -0.5673350, 0.6312910, -2.9354391, 3.3341267, -3.9014616, 3.5667300
4: -0.3886379, 0.6367542, -1.7549448, 3.5216486, -3.9102867, 2.3916986

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2979264, upper bound: 3.3149410
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3225889
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.9647372, 1.7185906, -1.0333478, 1.8749698, -2.8397064, 2.7519379
1: -1.5422183, 2.4143012, -1.6493901, 2.6237774, -4.1659951, 4.0636911
2: -1.0383445, 2.5049844, -1.1160958, 2.7230368, -3.7613811, 3.6210792
3: -2.7295728, 3.0798013, -2.9354391, 3.3341267, -6.0636997, 6.0152407
4: -1.6278296, 3.2467406, -1.7549448, 3.5216486, -5.1494780, 5.0016851

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2506352, upper bound: 3.3520483
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3895808
time: 0.30 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9253982, 1.6097733, -1.0172420, 1.7831979, -2.7085958, 2.6270151
1: -1.4741561, 2.2789021, -1.6082244, 2.5280578, -4.0022135, 3.8871264
2: -1.0221615, 2.3553617, -1.0998501, 2.5953715, -3.6175330, 3.4552112
3: -2.5333016, 2.9429610, -2.8071795, 3.2452543, -5.7785549, 5.7501402
4: -1.6486474, 3.0149312, -1.7616532, 3.3654399, -5.0140858, 4.7765841

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.28 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.57 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.2979264, upper bound: 3.2961051
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3040062
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.2908422, upper bound: 3.3611142
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.2881803, upper bound: 3.3628144
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.2979264, upper bound: 3.3149410
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3225889
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.2506352, upper bound: 3.3520483
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.3001019, upper bound: 3.3895808
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.1819023, 0.1728137, -0.2895984, 0.3943615, -0.5762638, 0.4624120
1: -0.2508491, 0.2474089, -0.4578271, 0.5641531, -0.8150022, 0.7052360
2: -0.2209612, 0.2683355, -0.3370600, 0.5814233, -0.8023845, 0.6053954
3: -0.3347811, 0.2938324, -0.6758766, 0.7399411, -1.0747222, 0.9697089
4: -0.2067872, 0.3344493, -0.4528955, 0.7513722, -0.9581594, 0.7873448

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2750512, upper bound: 3.2556361
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612091
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.1972761, 0.2204067, -0.2127345, 0.2789057, -0.4761818, 0.4331412
1: -0.2781399, 0.3030356, -0.3282925, 0.3944340, -0.6725738, 0.6313280
2: -0.2424032, 0.3270924, -0.2621886, 0.4136799, -0.6560830, 0.5892811
3: -0.3686350, 0.3737946, -0.4623349, 0.5059842, -0.8746191, 0.8361295
4: -0.2403007, 0.4049473, -0.3111475, 0.5232565, -0.7635573, 0.7160948

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2850107, upper bound: 3.2919480
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2824801, upper bound: 3.2814291
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.5662124, 0.8654976, -0.2895984, 0.3943615, -0.9605739, 1.1550959
1: -0.9263089, 1.2271175, -0.4578271, 0.5641531, -1.4904616, 1.6849444
2: -0.6173120, 1.3232578, -0.3370600, 0.5814233, -1.1987352, 1.6603178
3: -1.5797174, 1.6085356, -0.6758766, 0.7399411, -2.3196585, 2.2844121
4: -0.9259796, 1.7451435, -0.4528955, 0.7513722, -1.6773517, 2.1980391

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2841329, upper bound: 3.2903652
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3023597, upper bound: 3.3583707
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.5049574, 0.7416355, -0.2262982, 0.2959966, -0.8009540, 0.9679337
1: -0.8283286, 1.0456173, -0.3501132, 0.4226276, -1.2509562, 1.3957305
2: -0.5578265, 1.1398011, -0.2753200, 0.4363163, -0.9941428, 1.4151212
3: -1.3909674, 1.3781103, -0.4911644, 0.5441089, -1.9350762, 1.8692746
4: -0.8091390, 1.5109537, -0.3348299, 0.5523396, -1.3614784, 1.8457836

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3081144, upper bound: 3.3628145
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3081144, upper bound: 3.3628145
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1819023, 0.1728137, -1.0333478, 1.8749698, -2.0568721, 1.2061615
1: -0.2508491, 0.2474089, -1.6493901, 2.6237774, -2.8746264, 1.8967991
2: -0.2209612, 0.2683355, -1.1160958, 2.7230368, -2.9439979, 1.3844310
3: -0.3347811, 0.2938324, -2.9354391, 3.3341267, -3.6689076, 3.2292714
4: -0.2067872, 0.3344493, -1.7549448, 3.5216486, -3.7284358, 2.0893934

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551093, upper bound: 3.2834318
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.1972761, 0.2204067, -0.8552261, 1.4841180, -1.6813941, 1.0756326
1: -0.2781399, 0.3030356, -1.3730252, 2.0849776, -2.3631175, 1.6760607
2: -0.2424032, 0.3270924, -0.9147121, 2.1811352, -2.4235384, 1.2418046
3: -0.3686350, 0.3737946, -2.4193487, 2.6704021, -3.0390370, 2.7931433
4: -0.2403007, 0.4049473, -1.4258443, 2.8457990, -3.0860996, 1.8307915

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3621469, upper bound: 3.3023192
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.9647372, 1.7185906, -0.6319064, 0.9910424, -1.9557794, 2.3504958
1: -1.5422183, 2.4143012, -1.0185076, 1.3605185, -2.9027367, 3.4328089
2: -1.0383445, 2.5049844, -0.6755988, 1.5121008, -2.5504453, 3.1805828
3: -2.7295728, 3.0798013, -1.7817971, 1.7738161, -4.5033889, 4.8615985
4: -1.6278296, 3.2467406, -1.0166308, 1.9934199, -3.6212494, 4.2633715

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727990, upper bound: 3.3512972
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727990, upper bound: 3.3505638
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7944366, 1.3538276, -0.6699080, 1.0745010, -1.8689377, 2.0237355
1: -1.2789820, 1.8961051, -1.0852902, 1.5193026, -2.7982845, 2.9813950
2: -0.8491206, 2.0010822, -0.7177711, 1.6082748, -2.4573953, 2.7188532
3: -2.2459865, 2.4370046, -1.8827921, 1.9690655, -4.2150521, 4.3197966
4: -1.3163801, 2.6165221, -1.0920463, 2.1215045, -3.4378848, 3.7085683

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3277867, upper bound: 3.3741752
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3161056, upper bound: 3.3225462
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.9253982, 1.6097733, -0.4465413, 0.6192023, -1.5446005, 2.0563145
1: -1.4741561, 2.2789021, -0.7227822, 0.9138981, -2.3880544, 3.0016842
2: -1.0221615, 2.3553617, -0.4843388, 0.9321968, -1.9543581, 2.8397002
3: -2.5333016, 2.9429610, -1.1511729, 1.2243642, -3.7576656, 4.0941339
4: -1.6486474, 3.0149312, -0.7356169, 1.2473532, -2.8960006, 3.7505481

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9900093, upper bound: 3.0667480
time: 0.27 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.26 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.9253982, 1.6097733, -0.7372370, 1.2009584, -2.1263564, 2.3470104
1: -1.4741561, 2.2789021, -1.1906977, 1.6930063, -3.1671624, 3.4695997
2: -1.0221615, 2.3553617, -0.7919440, 1.7879090, -2.8100705, 3.1473057
3: -2.5333016, 2.9429610, -2.0395832, 2.2115228, -4.7448244, 4.9825439
4: -1.6486474, 3.0149312, -1.2595458, 2.3435612, -3.9922085, 4.2744770

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.27 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.60 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2750512, upper bound: 3.2556361
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612091
NS_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2850107, upper bound: 3.2919480
NS_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2824801, upper bound: 3.2814291
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2841329, upper bound: 3.2903652
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3023597, upper bound: 3.3583707
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3081144, upper bound: 3.3628145
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3081144, upper bound: 3.3628145
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3551093, upper bound: 3.2834318
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3621469, upper bound: 3.3023192
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2727990, upper bound: 3.3512972
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.2727990, upper bound: 3.3505638
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3277867, upper bound: 3.3741752
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.3161056, upper bound: 3.3225462
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
NS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.60
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1819023, 0.1728137, -0.1909689, 0.2010397, -0.3829420, 0.3637825
1: -0.2508491, 0.2474089, -0.2711346, 0.2899297, -0.5407789, 0.5185435
2: -0.2209612, 0.2683355, -0.2349898, 0.3038281, -0.5247893, 0.5033253
3: -0.3347811, 0.2938324, -0.3588729, 0.3552319, -0.6900129, 0.6527053
4: -0.2067872, 0.3344493, -0.2300957, 0.3767083, -0.5834955, 0.5645450

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2556361
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2556361
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1691999, 0.1290798, -0.1886021, 0.1795645, -0.3487643, 0.3176819
1: -0.2270265, 0.1939770, -0.2620578, 0.2599264, -0.4869529, 0.4560347
2: -0.2050433, 0.2022251, -0.2305741, 0.2687693, -0.4738125, 0.4327992
3: -0.3003190, 0.2287561, -0.3403661, 0.3140820, -0.6144010, 0.5691221
4: -0.1890953, 0.2461572, -0.2184048, 0.3308211, -0.5199164, 0.4645620

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612092
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612092
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.2127345, 0.2789057, -0.4511414, 0.3389110
1: -0.2309254, 0.1855326, -0.3282925, 0.3944340, -0.6253594, 0.5138251
2: -0.2092608, 0.1860420, -0.2621886, 0.4136799, -0.6229407, 0.4482306
3: -0.2956546, 0.2189267, -0.4623349, 0.5059842, -0.8016388, 0.6812616
4: -0.1927452, 0.2179434, -0.3111475, 0.5232565, -0.7160017, 0.5290909

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2905363
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1893517, 0.2139360, -0.3667888, 0.2625845
1: -0.2114542, 0.1079726, -0.2709337, 0.3011029, -0.5125571, 0.3789062
2: -0.1970211, 0.0937305, -0.2327694, 0.3237237, -0.5207448, 0.3264999
3: -0.2032855, 0.1348763, -0.3711154, 0.3736223, -0.5769079, 0.5059916
4: -0.1230521, 0.1147754, -0.2345843, 0.4059750, -0.5290272, 0.3493597

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743251
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.2895984, 0.3943615, -0.7051954, 0.7186424
1: -0.4992248, 0.5854943, -0.4578271, 0.5641531, -1.0633779, 1.0433213
2: -0.3621617, 0.6498277, -0.3370600, 0.5814233, -0.9435849, 0.9868875
3: -0.7706258, 0.7667233, -0.6758766, 0.7399411, -1.5105665, 1.4426000
4: -0.4680526, 0.8464483, -0.4528955, 0.7513722, -1.2194247, 1.2993438

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.2127345, 0.2789057, -0.6434696, 0.7205694
1: -0.5989561, 0.6957009, -0.3282925, 0.3944340, -0.9933901, 1.0239935
2: -0.4118725, 0.7733989, -0.2621886, 0.4136799, -0.8255523, 1.0355875
3: -0.9495392, 0.9206733, -0.4623349, 0.5059842, -1.4555233, 1.3830081
4: -0.5546308, 1.0178916, -0.3111475, 0.5232565, -1.0778873, 1.3290391

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583708
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5049574, 0.7416355, -0.1909689, 0.2010397, -0.7059971, 0.9326044
1: -0.8283286, 1.0456173, -0.2711346, 0.2899297, -1.1182584, 1.3167520
2: -0.5578265, 1.1398011, -0.2349898, 0.3038281, -0.8616546, 1.3747909
3: -1.3909674, 1.3781103, -0.3588729, 0.3552319, -1.7461993, 1.7369832
4: -0.8091390, 1.5109537, -0.2300957, 0.3767083, -1.1858473, 1.7410494

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2640640, upper bound: 3.2538683
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3030712, upper bound: 3.3612699
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5049574, 0.7416355, -0.1886021, 0.1795645, -0.6845219, 0.9302376
1: -0.8283286, 1.0456173, -0.2620578, 0.2599264, -1.0882550, 1.3076751
2: -0.5578265, 1.1398011, -0.2305741, 0.2687693, -0.8265958, 1.3703753
3: -1.3909674, 1.3781103, -0.3403661, 0.3140820, -1.7050494, 1.7184763
4: -0.8091390, 1.5109537, -0.2184048, 0.3308211, -1.1399601, 1.7293584

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2751297, upper bound: 3.3535318
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3030712, upper bound: 3.3612699
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1819023, 0.1728137, -0.6167870, 0.9804183, -1.1623206, 0.7896006
1: -0.2508491, 0.2474089, -1.0047641, 1.3907540, -1.6416031, 1.2521729
2: -0.2209612, 0.2683355, -0.6674653, 1.4859817, -1.7069429, 0.9358007
3: -0.3347811, 0.2938324, -1.7353394, 1.8086083, -2.1433892, 2.0291712
4: -0.2067872, 0.3344493, -1.0161959, 1.9481347, -2.1549218, 1.3506452

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1691999, 0.1290798, -0.5555521, 0.8390003, -1.0082002, 0.6846319
1: -0.2270265, 0.1939770, -0.9075407, 1.1878233, -1.4148498, 1.1015177
2: -0.2050433, 0.2022251, -0.6072665, 1.2878169, -1.4928601, 0.8094915
3: -0.3003190, 0.2287561, -1.5513695, 1.5564189, -1.8567379, 1.7801256
4: -0.1890953, 0.2461572, -0.8984371, 1.7021952, -1.8912905, 1.1445943

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551094, upper bound: 3.2834318
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551094, upper bound: 3.2834318
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1972761, 0.2204067, -0.4932721, 0.7089077, -0.9061838, 0.7136787
1: -0.2781399, 0.3030356, -0.8096205, 1.0052016, -1.2833414, 1.1126561
2: -0.2424032, 0.3270924, -0.5399144, 1.0911236, -1.3335267, 0.8670066
3: -0.3686350, 0.3737946, -1.3423414, 1.3339849, -1.7026198, 1.7161361
4: -0.2403007, 0.4049473, -0.7893663, 1.4534407, -1.6937414, 1.1943136

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1791718, 0.1613497, -0.4347970, 0.6191813, -0.7983531, 0.5961467
1: -0.2454292, 0.2264803, -0.7131633, 0.8658965, -1.1113257, 0.9396435
2: -0.2181719, 0.2404808, -0.4842769, 0.9492966, -1.1674685, 0.7247576
3: -0.3187436, 0.2694265, -1.1627244, 1.1476481, -1.4663917, 1.4321508
4: -0.2011959, 0.2918157, -0.6818090, 1.2631612, -1.4643571, 0.9736247

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619912, upper bound: 3.3007759
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991515, upper bound: 3.2851723
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.8381479, 1.4515730, -0.4437144, 0.6394674, -1.4776149, 1.8952874
1: -1.3446263, 2.0332325, -0.7238941, 0.8820208, -2.2266471, 2.7571266
2: -0.8959621, 2.1359978, -0.4936275, 0.9763694, -1.8723314, 2.6296253
3: -2.3707771, 2.6047986, -1.1795777, 1.1707634, -3.5415399, 3.7843759
4: -1.3932332, 2.7836850, -0.7038773, 1.3035572, -2.6967902, 3.4875622

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566218, upper bound: 3.2861896
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.9647372, 1.7185906, -0.5761735, 0.8669006, -1.8316377, 2.2947633
1: -1.5422183, 2.4143012, -0.9307029, 1.1916245, -2.7338428, 3.3450041
2: -1.0383445, 2.5049844, -0.6188431, 1.3351293, -2.3734736, 3.1238275
3: -2.7295728, 3.0798013, -1.6083661, 1.5656216, -4.2951941, 4.6881676
4: -1.6278296, 3.2467406, -0.9195610, 1.7740270, -3.4018564, 4.1663017

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3505478
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3505638
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8847823, 1.5582014, -0.6221162, 0.9677452, -1.8525274, 2.1803176
1: -1.4115896, 2.1846414, -1.0079430, 1.3673928, -2.7789824, 3.1925845
2: -0.9421095, 2.2875609, -0.6662498, 1.4640882, -2.4061978, 2.9538102
3: -2.5185604, 2.7853427, -1.7399702, 1.7814295, -4.2999897, 4.5253129
4: -1.4707451, 2.9702775, -1.0055420, 1.9379730, -3.4087181, 3.9758196

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.6210042, 0.9645910, -0.6699080, 1.0745010, -1.6955053, 1.6344991
1: -1.0059910, 1.3492717, -1.0852902, 1.5193026, -2.5252936, 2.4345617
2: -0.6661108, 1.4697034, -0.7177711, 1.6082748, -2.2743852, 2.1874745
3: -1.7382003, 1.7622229, -1.8827921, 1.9690655, -3.7072659, 3.6450145
4: -1.0016435, 1.9498672, -1.0920463, 2.1215045, -3.1231477, 3.0419133

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087359, upper bound: 3.3126019
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085308, upper bound: 3.3107573
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.3567759, 0.4924121, -1.3389742, 1.8234211
1: -1.3568091, 2.0639806, -0.5774629, 0.7274891, -2.0842981, 2.6414433
2: -0.9353622, 2.1358516, -0.3950991, 0.7278697, -1.6632320, 2.5309505
3: -2.3087072, 2.6659245, -0.8801971, 0.9667453, -3.2754524, 3.5461216
4: -1.4961721, 2.7389774, -0.5777845, 0.9593089, -2.4554811, 3.3167615

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.30 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.30 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.4465413, 0.6192023, -1.4879804, 1.9325418
1: -1.3866758, 2.1095767, -0.7227822, 0.9138981, -2.3005738, 2.8323588
2: -0.9571933, 2.1844273, -0.4843388, 0.9321968, -1.8893898, 2.6687658
3: -2.3729274, 2.7349393, -1.1511729, 1.2243642, -3.5972915, 3.8861122
4: -1.5417966, 2.8029537, -0.7356169, 1.2473532, -2.7891498, 3.5385706

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.29 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.7851678, 1.3140786, -0.4522020, 0.6309835, -1.4161510, 1.7662805
1: -1.2592278, 1.8705599, -0.7477506, 0.9067050, -2.1659327, 2.6183105
2: -0.8603663, 1.9465524, -0.4973092, 0.9481343, -1.8085005, 2.4438615
3: -2.1474485, 2.4361150, -1.1898680, 1.2277694, -3.3752179, 3.6259830
4: -1.3822207, 2.5080791, -0.7590128, 1.2772033, -2.6594241, 3.2670918

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.36 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.9253982, 1.6097733, -0.6832745, 1.0848455, -2.0102437, 2.2930479
1: -1.4741561, 2.2789021, -1.1057112, 1.5309575, -3.0051136, 3.3846133
2: -1.0221615, 2.3553617, -0.7312837, 1.6252620, -2.6474235, 3.0866451
3: -2.5333016, 2.9429610, -1.8844717, 2.0089293, -4.5422306, 4.8274326
4: -1.6486474, 3.0149312, -1.1575550, 2.1399906, -3.7886381, 4.1724854

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.30 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
time: 0.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.45 seconds
NS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2556361
NS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2556361
NS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612092
NS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2747273, upper bound: 3.2612092
NS_A1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
NS_A1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2905363
NS_A1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743251
NS_A1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
NS_A1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
NS_A1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
NS_A1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
NS_A1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583708
NS_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2640640, upper bound: 3.2538683
NS_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3030712, upper bound: 3.3612699
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2751297, upper bound: 3.3535318
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3030712, upper bound: 3.3612699
NS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3551094, upper bound: 3.2834318
NS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3551094, upper bound: 3.2834318
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3619912, upper bound: 3.3007759
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2991515, upper bound: 3.2851723
NS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2566218, upper bound: 3.2861896
NS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3505478
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3505638
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
NS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3087359, upper bound: 3.3126019
NS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3085308, upper bound: 3.3107573
NS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008
NS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.45
Output dim: 0, lower bound: -3.0033518, upper bound: 3.0844008

## BFS NS instance: NS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1909689, 0.2010397, -0.3534490, 0.2786164
1: -0.2129075, 0.1379279, -0.2711346, 0.2899297, -0.5028372, 0.4090625
2: -0.1962777, 0.1301043, -0.2349898, 0.3038281, -0.5001059, 0.3650941
3: -0.2156065, 0.1684411, -0.3588729, 0.3552319, -0.5708383, 0.5273141
4: -0.1264155, 0.1653319, -0.2300957, 0.3767083, -0.5031238, 0.3954276

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1909689, 0.2010397, -0.3557631, 0.2658362
1: -0.2135892, 0.1191348, -0.2711346, 0.2899297, -0.5035189, 0.3902693
2: -0.1987838, 0.1062041, -0.2349898, 0.3038281, -0.5026119, 0.3411939
3: -0.2093757, 0.1484058, -0.3588729, 0.3552319, -0.5646076, 0.5072787
4: -0.1251773, 0.1360584, -0.2300957, 0.3767083, -0.5018855, 0.3661542

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1886021, 0.1795645, -0.3319739, 0.2762496
1: -0.2129075, 0.1379279, -0.2620578, 0.2599264, -0.4728339, 0.3999857
2: -0.1962777, 0.1301043, -0.2305741, 0.2687693, -0.4650470, 0.3606784
3: -0.2156065, 0.1684411, -0.3403661, 0.3140820, -0.5296885, 0.5088072
4: -0.1264155, 0.1653319, -0.2184048, 0.3308211, -0.4572365, 0.3837367

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2562305
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2612091
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1886021, 0.1795645, -0.3342879, 0.2634695
1: -0.2135892, 0.1191348, -0.2620578, 0.2599264, -0.4735156, 0.3811925
2: -0.1987838, 0.1062041, -0.2305741, 0.2687693, -0.4675530, 0.3367783
3: -0.2093757, 0.1484058, -0.3403661, 0.3140820, -0.5234578, 0.4887719
4: -0.1251773, 0.1360584, -0.2184048, 0.3308211, -0.4559984, 0.3544632

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2538337
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2556361
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.1884287, 0.1961411, -0.3683768, 0.3146051
1: -0.2309254, 0.1855326, -0.2630137, 0.2788025, -0.5097278, 0.4485463
2: -0.2092608, 0.1860420, -0.2294976, 0.3010387, -0.5102995, 0.4155396
3: -0.2956546, 0.2189267, -0.3539879, 0.3374467, -0.6331013, 0.5729147
4: -0.1927452, 0.2179434, -0.2237305, 0.3769870, -0.5697322, 0.4416739

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.2079541, 0.2508155, -0.4230513, 0.3341305
1: -0.2309254, 0.1855326, -0.2982360, 0.3430882, -0.5740136, 0.4837686
2: -0.2092608, 0.1860420, -0.2566933, 0.3689277, -0.5781885, 0.4427353
3: -0.2956546, 0.2189267, -0.4100913, 0.4334176, -0.7290722, 0.6290181
4: -0.1927452, 0.2179434, -0.2668827, 0.4591578, -0.6519030, 0.4848261

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2904975
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2905363
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1726635, 0.1451107, -0.2979635, 0.2458963
1: -0.2114542, 0.1079726, -0.2342678, 0.2143725, -0.4258267, 0.3422404
2: -0.1970211, 0.0937305, -0.2095577, 0.2254317, -0.4224527, 0.3032882
3: -0.2032855, 0.1348763, -0.3108546, 0.2544775, -0.4577630, 0.4457309
4: -0.1230521, 0.1147754, -0.1944999, 0.2772832, -0.4003353, 0.3092753

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743115
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743115
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1858697, 0.1883028, -0.3411556, 0.2591025
1: -0.2114542, 0.1079726, -0.2582642, 0.2605539, -0.4720081, 0.3662368
2: -0.1970211, 0.0937305, -0.2268859, 0.2786751, -0.4756962, 0.3206164
3: -0.2032855, 0.1348763, -0.3372544, 0.3132812, -0.5165668, 0.4721307
4: -0.1230521, 0.1147754, -0.2129510, 0.3409672, -0.4640194, 0.3277264

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.1884287, 0.1961411, -0.5069750, 0.6174728
1: -0.4992248, 0.5854943, -0.2630137, 0.2788025, -0.7780271, 0.8485080
2: -0.3621617, 0.6498277, -0.2294976, 0.3010387, -0.6632003, 0.8793253
3: -0.7706258, 0.7667233, -0.3539879, 0.3374467, -1.1080723, 1.1207112
4: -0.4680526, 0.8464483, -0.2237305, 0.3769870, -0.8450395, 1.0701789

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.2079541, 0.2508155, -0.5616494, 0.6369982
1: -0.4992248, 0.5854943, -0.2982360, 0.3430882, -0.8423130, 0.8837304
2: -0.3621617, 0.6498277, -0.2566933, 0.3689277, -0.7310895, 0.9065210
3: -0.7706258, 0.7667233, -0.4100913, 0.4334176, -1.2040433, 1.1768147
4: -0.4680526, 0.8464483, -0.2668827, 0.4591578, -0.9272104, 1.1133311

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.1884287, 0.1961411, -0.5607051, 0.6962636
1: -0.5989561, 0.6957009, -0.2630137, 0.2788025, -0.8777585, 0.9587146
2: -0.4118725, 0.7733989, -0.2294976, 0.3010387, -0.7129110, 1.0028965
3: -0.9495392, 0.9206733, -0.3539879, 0.3374467, -1.2869859, 1.2746612
4: -0.5546308, 1.0178916, -0.2237305, 0.3769870, -0.9316177, 1.2416222

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.2079541, 0.2508155, -0.6153795, 0.7157890
1: -0.5989561, 0.6957009, -0.2982360, 0.3430882, -0.9420443, 0.9939370
2: -0.4118725, 0.7733989, -0.2566933, 0.3689277, -0.7808002, 1.0300922
3: -0.9495392, 0.9206733, -0.4100913, 0.4334176, -1.3829567, 1.3307645
4: -0.5546308, 1.0178916, -0.2668827, 0.4591578, -1.0137885, 1.2847744

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583707
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583708
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2652147, 0.3477954, -0.1909689, 0.2010397, -0.4662544, 0.5387643
1: -0.4077931, 0.4683562, -0.2711346, 0.2899297, -0.6977229, 0.7394907
2: -0.3259546, 0.5277624, -0.2349898, 0.3038281, -0.6297828, 0.7627522
3: -0.6094154, 0.6039184, -0.3588729, 0.3552319, -0.9646472, 0.9627913
4: -0.3659086, 0.6728157, -0.2300957, 0.3767083, -0.7426169, 0.9029114

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2693666, 0.3691269, -0.1697195, 0.1387336, -0.4081001, 0.5388464
1: -0.4349672, 0.5040386, -0.2324924, 0.2061463, -0.6411136, 0.7365309
2: -0.3306690, 0.5494190, -0.2065555, 0.2129451, -0.5436141, 0.7559745
3: -0.6472651, 0.6534121, -0.3027957, 0.2431646, -0.8904297, 0.9562078
4: -0.3854734, 0.6977538, -0.1897042, 0.2581896, -0.6436629, 0.8874580

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2775669, upper bound: 3.3543547
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2775669, upper bound: 3.3619991
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5049574, 0.7416355, -0.1555604, 0.0857961, -0.5907535, 0.8971959
1: -0.8283286, 1.0456173, -0.2158905, 0.1343637, -0.9626923, 1.2615077
2: -0.5578265, 1.1398011, -0.2000806, 0.1217305, -0.6795570, 1.3398817
3: -1.3909674, 1.3781103, -0.2145249, 0.1663484, -1.5573155, 1.5926352
4: -0.8091390, 1.5109537, -0.1282436, 0.1528131, -0.9619520, 1.6391972

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2747888, upper bound: 3.3535018
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2609166, upper bound: 3.2955470
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3932320, 0.5561796, -0.1538953, 0.0847333, -0.4779653, 0.7100749
1: -0.6459870, 0.7728406, -0.2140015, 0.1236971, -0.7696842, 0.9868422
2: -0.4441655, 0.8514178, -0.1986294, 0.1090919, -0.5532575, 1.0500472
3: -1.0361072, 1.0229441, -0.2086054, 0.1530888, -1.1891956, 1.2315495
4: -0.6119018, 1.1262436, -0.1261868, 0.1310454, -0.7429473, 1.2524304

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008124, upper bound: 3.3610653
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2859449, upper bound: 3.3002928
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.6167870, 0.9804183, -1.1328278, 0.7044343
1: -0.2129075, 0.1379279, -1.0047641, 1.3907540, -1.6036614, 1.1426920
2: -0.1962777, 0.1301043, -0.6674653, 1.4859817, -1.6822594, 0.7975696
3: -0.2156065, 0.1684411, -1.7353394, 1.8086083, -2.0242147, 1.9037805
4: -0.1264155, 0.1653319, -1.0161959, 1.9481347, -2.0745502, 1.1815277

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.6167870, 0.9804183, -1.1351418, 0.6916543
1: -0.2135892, 0.1191348, -1.0047641, 1.3907540, -1.6043432, 1.1238989
2: -0.1987838, 0.1062041, -0.6674653, 1.4859817, -1.6847655, 0.7736694
3: -0.2093757, 0.1484058, -1.7353394, 1.8086083, -2.0179839, 1.8837447
4: -0.1251773, 0.1360584, -1.0161959, 1.9481347, -2.0733120, 1.1522542

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.5555521, 0.8390003, -0.9914097, 0.6431996
1: -0.2129075, 0.1379279, -0.9075407, 1.1878233, -1.4007307, 1.0454686
2: -0.1962777, 0.1301043, -0.6072665, 1.2878169, -1.4840946, 0.7373707
3: -0.2156065, 0.1684411, -1.5513695, 1.5564189, -1.7720253, 1.7198107
4: -0.1264155, 0.1653319, -0.8984371, 1.7021952, -1.8286107, 1.0637690

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2834318
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2834316
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.5555521, 0.8390003, -0.9937239, 0.6304194
1: -0.2135892, 0.1191348, -0.9075407, 1.1878233, -1.4014125, 1.0266751
2: -0.1987838, 0.1062041, -0.6072665, 1.2878169, -1.4866006, 0.7134706
3: -0.2093757, 0.1484058, -1.5513695, 1.5564189, -1.7657946, 1.6997753
4: -0.1251773, 0.1360584, -0.8984371, 1.7021952, -1.8273724, 1.0344956

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.4932721, 0.7089077, -0.8811434, 0.6194484
1: -0.2309254, 0.1855326, -0.8096205, 1.0052016, -1.2361269, 0.9951531
2: -0.2092608, 0.1860420, -0.5399144, 1.0911236, -1.3003844, 0.7259563
3: -0.2956546, 0.2189267, -1.3423414, 1.3339849, -1.6296395, 1.5612682
4: -0.1927452, 0.2179434, -0.7893663, 1.4534407, -1.6461859, 1.0073097

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.4932721, 0.7089077, -0.8617605, 0.5665048
1: -0.2114542, 0.1079726, -0.8096205, 1.0052016, -1.2166557, 0.9175931
2: -0.1970211, 0.0937305, -0.5399144, 1.0911236, -1.2881447, 0.6336449
3: -0.2032855, 0.1348763, -1.3423414, 1.3339849, -1.5372704, 1.4772177
4: -0.1230521, 0.1147754, -0.7893663, 1.4534407, -1.5764928, 0.9041417

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2938095, upper bound: 3.2833261
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1754401, 0.1463295, -0.4540939, 0.6571444, -0.8325846, 0.6004235
1: -0.2380556, 0.2082127, -0.7423609, 0.9204669, -1.1585226, 0.9505736
2: -0.2130748, 0.2189502, -0.5018728, 0.9998336, -1.2129084, 0.7208230
3: -0.3082939, 0.2473288, -1.2258428, 1.2148058, -1.5230997, 1.4731716
4: -0.1959600, 0.2641106, -0.7120590, 1.3295162, -1.5254762, 0.9761696

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2860204
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1791718, 0.1613497, -0.3292880, 0.4606968, -0.6398686, 0.4906377
1: -0.2454292, 0.2264803, -0.5379574, 0.6420698, -0.8874990, 0.7644376
2: -0.2181719, 0.2404808, -0.3788580, 0.6950080, -0.9131799, 0.6193388
3: -0.3187436, 0.2694265, -0.8339219, 0.8400441, -1.1587877, 1.1033485
4: -0.2011959, 0.2918157, -0.4960041, 0.9036072, -1.1048031, 0.7878196

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2567721
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2804862
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4855508, 0.6990324, -0.4437144, 0.6394674, -1.1250178, 1.1427468
1: -0.7859744, 0.9459003, -0.7238941, 0.8820208, -1.6679952, 1.6697944
2: -0.5284564, 1.0867116, -0.4936275, 0.9763694, -1.5048258, 1.5803392
3: -1.3301110, 1.2562753, -1.1795777, 1.1707634, -2.5008743, 2.4358528
4: -0.7626743, 1.4561044, -0.7038773, 1.3035572, -2.0662315, 2.1599813

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2725631
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2861896
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5079609, 0.7406304, -0.4437144, 0.6394674, -1.1474280, 1.1843446
1: -0.8270459, 1.0331752, -0.7238941, 0.8820208, -1.7090667, 1.7570693
2: -0.5495639, 1.1436493, -0.4936275, 0.9763694, -1.5259334, 1.6372769
3: -1.4009477, 1.3651351, -1.1795777, 1.1707634, -2.5717108, 2.5447128
4: -0.8036790, 1.5224344, -0.7038773, 1.3035572, -2.1072361, 2.2263110

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7350374, 1.2556964, -0.5761735, 0.8669006, -1.6019380, 1.8318698
1: -1.1878364, 1.7758900, -0.9307029, 1.1916245, -2.3794608, 2.7065926
2: -0.7947478, 1.8467489, -0.6188431, 1.3351293, -2.1298771, 2.4655919
3: -2.0574775, 2.2811511, -1.6083661, 1.5656216, -3.6230991, 3.8895168
4: -1.2354655, 2.4035592, -0.9195610, 1.7740270, -3.0094924, 3.3231199

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2434433, upper bound: 3.2618185
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2654457, upper bound: 3.3445908
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.8940496, 1.5659399, -0.5761735, 0.8669006, -1.7609501, 2.1421134
1: -1.4317443, 2.2020078, -0.9307029, 1.1916245, -2.6233687, 3.1327105
2: -0.9584477, 2.2942798, -0.6188431, 1.3351293, -2.2935770, 2.9131227
3: -2.5250781, 2.8155832, -1.6083661, 1.5656216, -4.0906997, 4.4239492
4: -1.4964833, 2.9824798, -0.9195610, 1.7740270, -3.2705102, 3.9020405

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2434433, upper bound: 3.3300085
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2654457, upper bound: 3.3445908
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.7774747, 1.3240056, -0.5139025, 0.7664561, -1.5439308, 1.8379080
1: -1.2421060, 1.8495609, -0.8436866, 1.0788965, -2.3210020, 2.6932476
2: -0.8266892, 1.9650905, -0.5637228, 1.1745249, -2.0012138, 2.5288129
3: -2.2023203, 2.3728528, -1.4127698, 1.4236665, -3.6259866, 3.7856226
4: -1.2790440, 2.5647111, -0.8300856, 1.5632312, -2.8422751, 3.3947966

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8847823, 1.5582014, -0.5583415, 0.8277088, -1.7124910, 2.1165428
1: -1.4115896, 2.1846414, -0.9042768, 1.1698270, -2.5814161, 3.0889177
2: -0.9421095, 2.2875609, -0.5974064, 1.2688980, -2.2110076, 2.8849671
3: -2.5185604, 2.7853427, -1.5432237, 1.5357510, -4.0543103, 4.3285651
4: -1.4707451, 2.9702775, -0.8885157, 1.6892197, -3.1599648, 3.8587933

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707630
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3245428, 0.4519358, -0.6699080, 1.0745010, -1.3990438, 1.1218438
1: -0.5311444, 0.6332443, -1.0852902, 1.5193026, -2.0504470, 1.7185345
2: -0.3751126, 0.6765136, -0.7177711, 1.6082748, -1.9833871, 1.3942846
3: -0.8123156, 0.8308139, -1.8827921, 1.9690655, -2.7813811, 2.7136059
4: -0.4943081, 0.8798682, -1.0920463, 2.1215045, -2.6158128, 1.9719145

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3107573
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.2881652, 0.3994725, -0.5468583, 0.8083949, -1.0965596, 0.9463308
1: -0.4703782, 0.5547249, -0.8886769, 1.1410204, -1.6113986, 1.4434018
2: -0.3435284, 0.5959076, -0.5877235, 1.2417006, -1.5852292, 1.1836307
3: -0.7070373, 0.7215918, -1.5159041, 1.5002446, -2.2072821, 2.2374957
4: -0.4266806, 0.7662221, -0.8726943, 1.6501126, -2.0767932, 1.6389161

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3107573
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.3591895, 0.5100847, -1.3566468, 1.8258346
1: -1.3568091, 2.0639806, -0.5876420, 0.7416234, -2.0984323, 2.6516225
2: -0.9353622, 2.1358516, -0.4022615, 0.7563208, -1.6916831, 2.5381131
3: -2.3087072, 2.6659245, -0.9060956, 0.9887789, -3.2974863, 3.5720201
4: -1.4961721, 2.7389774, -0.5930994, 1.0039839, -2.5001557, 3.3320768

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
time: 0.32 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.4034397, 0.5544606, -1.4010228, 1.8700852
1: -1.3568091, 2.0639806, -0.6527545, 0.8233188, -2.1801279, 2.7167351
2: -0.9353622, 2.1358516, -0.4403477, 0.8237113, -1.7590735, 2.5761993
3: -2.3087072, 2.6659245, -1.0165187, 1.0986658, -3.4073730, 3.6824431
4: -1.4961721, 2.7389774, -0.6588401, 1.0977876, -2.5939593, 3.3978167

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
time: 0.33 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.3591895, 0.5100847, -1.3788629, 1.8451900
1: -1.3866758, 2.1095767, -0.5876420, 0.7416234, -2.1282992, 2.6972182
2: -0.9571933, 2.1844273, -0.4022615, 0.7563208, -1.7135139, 2.5866888
3: -2.3729274, 2.7349393, -0.9060956, 0.9887789, -3.3617063, 3.6410351
4: -1.5417966, 2.8029537, -0.5930994, 1.0039839, -2.5457802, 3.3960531

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.33 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.33 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.4037730, 0.5570420, -1.4258202, 1.8897735
1: -1.3866758, 2.1095767, -0.6538737, 0.8274131, -2.2140889, 2.7634504
2: -0.9571933, 2.1844273, -0.4414503, 0.8264231, -1.7836161, 2.6258776
3: -2.3729274, 2.7349393, -1.0185462, 1.1043648, -3.4772916, 3.7534857
4: -1.5417966, 2.8029537, -0.6610875, 1.1015793, -2.6433759, 3.4640408

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.31 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
time: 0.33 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8364059, 1.4454074, -0.4522020, 0.6309835, -1.4673891, 1.8976094
1: -1.3392069, 2.0378947, -0.7477506, 0.9067050, -2.2459121, 2.7856455
2: -0.9222740, 2.1100194, -0.4973092, 0.9481343, -1.8704083, 2.6073284
3: -2.2800412, 2.6337733, -1.1898680, 1.2277694, -3.5078104, 3.8236413
4: -1.4792006, 2.6968720, -0.7590128, 1.2772033, -2.7564037, 3.4558847

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.32 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.8633735, 1.4755141, -0.4522020, 0.6309835, -1.4943569, 1.9277158
1: -1.3774875, 2.0958517, -0.7477506, 0.9067050, -2.2841921, 2.8436022
2: -0.9503216, 2.1727619, -0.4973092, 0.9481343, -1.8984559, 2.6700709
3: -2.3591931, 2.7177744, -1.1898680, 1.2277694, -3.5869625, 3.9076424
4: -1.5330763, 2.7828827, -0.7590128, 1.2772033, -2.8102791, 3.5418954

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.32 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.6832745, 1.0848455, -1.9314077, 2.1499200
1: -1.3568091, 2.0639806, -1.1057112, 1.5309575, -2.8877661, 3.1696913
2: -0.9353622, 2.1358516, -0.7312837, 1.6252620, -2.5606241, 2.8671350
3: -2.3087072, 2.6659245, -1.8844717, 2.0089293, -4.3176365, 4.5503960
4: -1.4961721, 2.7389774, -1.1575550, 2.1399906, -3.6361628, 3.8965309

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0844008
time: 0.34 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0844008
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.6832745, 1.0848455, -1.9536238, 2.1692750
1: -1.3866758, 2.1095767, -1.1057112, 1.5309575, -2.9176333, 3.2152877
2: -0.9571933, 2.1844273, -0.7312837, 1.6252620, -2.5824552, 2.9157104
3: -2.3729274, 2.7349393, -1.8844717, 2.0089293, -4.3818569, 4.6194110
4: -1.5417966, 2.8029537, -1.1575550, 2.1399906, -3.6817870, 3.9605081

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0794402
time: 0.35 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0794402
time: 0.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.18 seconds
NS_A1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
NS_A1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
NS_A1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
NS_A1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
NS_A1_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2562305
NS_A1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2612091
NS_A1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2538337
NS_A1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2497337, upper bound: 3.2556361
NS_A1_B1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
NS_A1_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
NS_A1_B1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2904975
NS_A1_B1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2905363
NS_A1_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743115
NS_A1_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2743115
NS_A1_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
NS_A1_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2545485, upper bound: 3.2813351
NS_A1_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
NS_A1_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2832706
NS_A1_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
NS_A1_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2574151, upper bound: 3.2903652
NS_A1_B1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
NS_A1_B1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3498894
NS_A1_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583707
NS_A1_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2742534, upper bound: 3.3583708
NS_A1_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
NS_A1_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
NS_A1_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2775669, upper bound: 3.3543547
NS_A1_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2775669, upper bound: 3.3619991
NS_A1_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2747888, upper bound: 3.3535018
NS_A1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2609166, upper bound: 3.2955470
NS_A1_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3008124, upper bound: 3.3610653
NS_A1_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2859449, upper bound: 3.3002928
NS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2834318
NS_A1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2834316
NS_A1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2852605, upper bound: 3.2557639
NS_A1_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
NS_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3045499, upper bound: 3.2872347
NS_A1_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
NS_A1_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2938095, upper bound: 3.2833261
NS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
NS_A1_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2860204
NS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2567721
NS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2804862
NS_A1_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2725631
NS_A1_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2861896
NS_A1_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
NS_A1_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2566218, upper bound: 3.3512972
NS_A1_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2434433, upper bound: 3.2618185
NS_A1_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2654457, upper bound: 3.3445908
NS_A1_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2434433, upper bound: 3.3300085
NS_A1_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2654457, upper bound: 3.3445908
NS_A1_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
NS_A1_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3276017, upper bound: 3.3724500
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707630
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
NS_A1_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
NS_A1_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3107573
NS_A1_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
NS_A1_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3107573
NS_A2_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
NS_A2_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
NS_A2_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
NS_A2_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3896346, upper bound: 3.3926028
NS_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -3.3892228, upper bound: 3.3892228
NS_A2_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0697922
NS_A2_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0844008
NS_A2_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0844008
NS_A2_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0794402
NS_A2_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.9916026, upper bound: 3.0794402

## BFS NS instance: NS_A1_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1640568, 0.0970304, -0.2494397, 0.2517043
1: -0.2129075, 0.1379279, -0.2160547, 0.1573067, -0.3702142, 0.3539826
2: -0.1962777, 0.1301043, -0.1979575, 0.1493119, -0.3455896, 0.3280618
3: -0.2156065, 0.1684411, -0.2790203, 0.1848386, -0.4004451, 0.4474615
4: -0.1264155, 0.1653319, -0.1823387, 0.1798168, -0.3062323, 0.3476706

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580574, upper bound: 3.2784066
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580574, upper bound: 3.2784066
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1756259, 0.1411821, -0.2935914, 0.2632734
1: -0.2129075, 0.1379279, -0.2377549, 0.2046857, -0.4175932, 0.3756828
2: -0.1962777, 0.1301043, -0.2134545, 0.2086003, -0.4048780, 0.3435588
3: -0.2156065, 0.1684411, -0.3056127, 0.2426994, -0.4583058, 0.4740539
4: -0.1264155, 0.1653319, -0.1975847, 0.2468585, -0.3732740, 0.3629166

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580574, upper bound: 3.2854497
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2580574, upper bound: 3.2854497
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1640568, 0.0970304, -0.2517539, 0.2389242
1: -0.2135892, 0.1191348, -0.2160547, 0.1573067, -0.3708959, 0.3351895
2: -0.1987838, 0.1062041, -0.1979575, 0.1493119, -0.3480957, 0.3041616
3: -0.2093757, 0.1484058, -0.2790203, 0.1848386, -0.3942143, 0.4274261
4: -0.1251773, 0.1360584, -0.1823387, 0.1798168, -0.3049941, 0.3183971

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2538337
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1756259, 0.1411821, -0.2959055, 0.2504932
1: -0.2135892, 0.1191348, -0.2377549, 0.2046857, -0.4182749, 0.3568897
2: -0.1987838, 0.1062041, -0.2134545, 0.2086003, -0.4073841, 0.3196586
3: -0.2093757, 0.1484058, -0.3056127, 0.2426994, -0.4520751, 0.4540185
4: -0.1251773, 0.1360584, -0.1975847, 0.2468585, -0.3720358, 0.3336432

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505724, upper bound: 3.2556361
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1555604, 0.0857961, -0.2382055, 0.2432079
1: -0.2129075, 0.1379279, -0.2158905, 0.1343637, -0.3472712, 0.3538184
2: -0.1962777, 0.1301043, -0.2000806, 0.1217305, -0.3180082, 0.3301848
3: -0.2156065, 0.1684411, -0.2145249, 0.1663484, -0.3819548, 0.3829660
4: -0.1264155, 0.1653319, -0.1282436, 0.1528131, -0.2792285, 0.2935754

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2551363, upper bound: 3.2750599
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2551363, upper bound: 3.2750599
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.1538953, 0.0847333, -0.2371427, 0.2415428
1: -0.2129075, 0.1379279, -0.2140015, 0.1236971, -0.3366046, 0.3519295
2: -0.1962777, 0.1301043, -0.1986294, 0.1090919, -0.3053696, 0.3287337
3: -0.2156065, 0.1684411, -0.2086054, 0.1530888, -0.3686953, 0.3770465
4: -0.1264155, 0.1653319, -0.1261868, 0.1310454, -0.2574609, 0.2915187

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2551363, upper bound: 3.2852095
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2551363, upper bound: 3.2852095
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1555604, 0.0857961, -0.2405196, 0.2304277
1: -0.2135892, 0.1191348, -0.2158905, 0.1343637, -0.3479529, 0.3350253
2: -0.1987838, 0.1062041, -0.2000806, 0.1217305, -0.3205143, 0.3062847
3: -0.2093757, 0.1484058, -0.2145249, 0.1663484, -0.3757241, 0.3629307
4: -0.1251773, 0.1360584, -0.1282436, 0.1528131, -0.2779903, 0.2643020

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2466018, upper bound: 3.2475665
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2480080, upper bound: 3.2516749
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.1538953, 0.0847333, -0.2394568, 0.2287626
1: -0.2135892, 0.1191348, -0.2140015, 0.1236971, -0.3372863, 0.3331363
2: -0.1987838, 0.1062041, -0.1986294, 0.1090919, -0.3078757, 0.3048335
3: -0.2093757, 0.1484058, -0.2086054, 0.1530888, -0.3624645, 0.3570112
4: -0.1251773, 0.1360584, -0.1261868, 0.1310454, -0.2562227, 0.2622453

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2466018, upper bound: 3.2498040
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2480080, upper bound: 3.2520055
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.1640568, 0.0970304, -0.2692661, 0.2902333
1: -0.2309254, 0.1855326, -0.2160547, 0.1573067, -0.3882321, 0.4015872
2: -0.2092608, 0.1860420, -0.1979575, 0.1493119, -0.3585728, 0.3839995
3: -0.2956546, 0.2189267, -0.2790203, 0.1848386, -0.4804932, 0.4979470
4: -0.1927452, 0.2179434, -0.1823387, 0.1798168, -0.3725621, 0.4002821

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2808860
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.1555604, 0.0857961, -0.2580318, 0.2817368
1: -0.2309254, 0.1855326, -0.2158905, 0.1343637, -0.3652891, 0.4014230
2: -0.2092608, 0.1860420, -0.2000806, 0.1217305, -0.3309914, 0.3861226
3: -0.2956546, 0.2189267, -0.2145249, 0.1663484, -0.4620030, 0.4334516
4: -0.1927452, 0.2179434, -0.1282436, 0.1528131, -0.3455583, 0.3461869

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2566950, upper bound: 3.2816074
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2303078, upper bound: 3.2743419
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.1756259, 0.1411821, -0.3134178, 0.3018023
1: -0.2309254, 0.1855326, -0.2377549, 0.2046857, -0.4356111, 0.4232875
2: -0.2092608, 0.1860420, -0.2134545, 0.2086003, -0.4178611, 0.3994965
3: -0.2956546, 0.2189267, -0.3056127, 0.2426994, -0.5383540, 0.5245395
4: -0.1927452, 0.2179434, -0.1975847, 0.2468585, -0.4396037, 0.4155281

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831822, upper bound: 3.2904975
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831822, upper bound: 3.2904975
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.1538953, 0.0847333, -0.2569691, 0.2800717
1: -0.2309254, 0.1855326, -0.2140015, 0.1236971, -0.3546225, 0.3995341
2: -0.2092608, 0.1860420, -0.1986294, 0.1090919, -0.3183528, 0.3846714
3: -0.2956546, 0.2189267, -0.2086054, 0.1530888, -0.4487434, 0.4275321
4: -0.1927452, 0.2179434, -0.1261868, 0.1310454, -0.3237907, 0.3441302

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2804253, upper bound: 3.2905363
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831821, upper bound: 3.2905362
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831822, upper bound: 3.2905363
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1640568, 0.0970304, -0.2498832, 0.2372896
1: -0.2114542, 0.1079726, -0.2160547, 0.1573067, -0.3687609, 0.3240273
2: -0.1970211, 0.0937305, -0.1979575, 0.1493119, -0.3463330, 0.2916880
3: -0.2032855, 0.1348763, -0.2790203, 0.1848386, -0.3881242, 0.4138966
4: -0.1230521, 0.1147754, -0.1823387, 0.1798168, -0.3028690, 0.2971141

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2539935, upper bound: 3.2703697
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2291571, upper bound: 3.2632039
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1555604, 0.0857961, -0.2386489, 0.2287932
1: -0.2114542, 0.1079726, -0.2158905, 0.1343637, -0.3458179, 0.3238630
2: -0.1970211, 0.0937305, -0.2000806, 0.1217305, -0.3187516, 0.2938111
3: -0.2032855, 0.1348763, -0.2145249, 0.1663484, -0.3696339, 0.3494012
4: -0.1230521, 0.1147754, -0.1282436, 0.1528131, -0.2758652, 0.2430189

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2539935, upper bound: 3.2703697
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2291571, upper bound: 3.2655633
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1756259, 0.1411821, -0.2940348, 0.2488587
1: -0.2114542, 0.1079726, -0.2377549, 0.2046857, -0.4161399, 0.3457275
2: -0.1970211, 0.0937305, -0.2134545, 0.2086003, -0.4056214, 0.3071851
3: -0.2032855, 0.1348763, -0.3056127, 0.2426994, -0.4459849, 0.4404890
4: -0.1230521, 0.1147754, -0.1975847, 0.2468585, -0.3699106, 0.3123601

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2813351, upper bound: 3.2813351
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2813351, upper bound: 3.2813351
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.1538953, 0.0847333, -0.2375861, 0.2271281
1: -0.2114542, 0.1079726, -0.2140015, 0.1236971, -0.3351513, 0.3219741
2: -0.1970211, 0.0937305, -0.1986294, 0.1090919, -0.3061130, 0.2923599
3: -0.2032855, 0.1348763, -0.2086054, 0.1530888, -0.3563743, 0.3434817
4: -0.1230521, 0.1147754, -0.1261868, 0.1310454, -0.2540976, 0.2409622

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2732609, upper bound: 3.2754562
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2813351, upper bound: 3.2813351
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2813351, upper bound: 3.2813351
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.1640568, 0.0970304, -0.4078643, 0.5931009
1: -0.4992248, 0.5854943, -0.2160547, 0.1573067, -0.6565315, 0.8015490
2: -0.3621617, 0.6498277, -0.1979575, 0.1493119, -0.5114736, 0.8477852
3: -0.7706258, 0.7667233, -0.2790203, 0.1848386, -0.9554642, 1.0457437
4: -0.4680526, 0.8464483, -0.1823387, 0.1798168, -0.6478694, 1.0287870

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2273866, upper bound: 3.2568519
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2325686, upper bound: 3.2748992
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.1555604, 0.0857961, -0.3966300, 0.5846044
1: -0.4992248, 0.5854943, -0.2158905, 0.1343637, -0.6335885, 0.8013848
2: -0.3621617, 0.6498277, -0.2000806, 0.1217305, -0.4838923, 0.8499083
3: -0.7706258, 0.7667233, -0.2145249, 0.1663484, -0.9369740, 0.9812482
4: -0.4680526, 0.8464483, -0.1282436, 0.1528131, -0.6208656, 0.9746919

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2273866, upper bound: 3.2642130
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2325686, upper bound: 3.2821104
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.1756259, 0.1411821, -0.4520160, 0.6046699
1: -0.4992248, 0.5854943, -0.2377549, 0.2046857, -0.7039105, 0.8232492
2: -0.3621617, 0.6498277, -0.2134545, 0.2086003, -0.5707619, 0.8632822
3: -0.7706258, 0.7667233, -0.3056127, 0.2426994, -1.0133249, 1.0723361
4: -0.4680526, 0.8464483, -0.1975847, 0.2468585, -0.7149110, 1.0440331

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2840614, upper bound: 3.2903652
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2840614, upper bound: 3.2903652
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.3108339, 0.4290441, -0.1538953, 0.0847333, -0.3955672, 0.5829393
1: -0.4992248, 0.5854943, -0.2140015, 0.1236971, -0.6229219, 0.7994959
2: -0.3621617, 0.6498277, -0.1986294, 0.1090919, -0.4712536, 0.8484571
3: -0.7706258, 0.7667233, -0.2086054, 0.1530888, -0.9237146, 0.9753287
4: -0.4680526, 0.8464483, -0.1261868, 0.1310454, -0.5990980, 0.9726351

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2840614, upper bound: 3.2903652
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2840614, upper bound: 3.2903652
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.1640568, 0.0970304, -0.4615943, 0.6718917
1: -0.5989561, 0.6957009, -0.2160547, 0.1573067, -0.7562627, 0.9117556
2: -0.4118725, 0.7733989, -0.1979575, 0.1493119, -0.5611844, 0.9713565
3: -0.9495392, 0.9206733, -0.2790203, 0.1848386, -1.1343775, 1.1996936
4: -0.5546308, 1.0178916, -0.1823387, 0.1798168, -0.7344477, 1.2002304

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742326, upper bound: 3.3494437
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2490266, upper bound: 3.3394521
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.1555604, 0.0857961, -0.4503600, 0.6633953
1: -0.5989561, 0.6957009, -0.2158905, 0.1343637, -0.7333198, 0.9115914
2: -0.4118725, 0.7733989, -0.2000806, 0.1217305, -0.5336030, 0.9734795
3: -0.9495392, 0.9206733, -0.2145249, 0.1663484, -1.1158872, 1.1351981
4: -0.5546308, 1.0178916, -0.1282436, 0.1528131, -0.7074439, 1.1461352

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2742322, upper bound: 3.3494436
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2490266, upper bound: 3.3466722
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.1756259, 0.1411821, -0.5057460, 0.6834608
1: -0.5989561, 0.6957009, -0.2377549, 0.2046857, -0.8036419, 0.9334558
2: -0.4118725, 0.7733989, -0.2134545, 0.2086003, -0.6204726, 0.9868535
3: -0.9495392, 0.9206733, -0.3056127, 0.2426994, -1.1922383, 1.2262859
4: -0.5546308, 1.0178916, -0.1975847, 0.2468585, -0.8014892, 1.2154764

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001008, upper bound: 3.3581531
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2863453, upper bound: 3.3034404
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3645640, 0.5078349, -0.1538953, 0.0847333, -0.4492972, 0.6617302
1: -0.5989561, 0.6957009, -0.2140015, 0.1236971, -0.7226533, 0.9097025
2: -0.4118725, 0.7733989, -0.1986294, 0.1090919, -0.5209644, 0.9720283
3: -0.9495392, 0.9206733, -0.2086054, 0.1530888, -1.1026275, 1.1292787
4: -0.5546308, 1.0178916, -0.1261868, 0.1310454, -0.6856763, 1.1440785

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012147, upper bound: 3.3582767
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012147, upper bound: 3.3583707
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2652147, 0.3477954, -0.1640568, 0.0970304, -0.3622451, 0.5118523
1: -0.4077931, 0.4683562, -0.2160547, 0.1573067, -0.5650998, 0.6844109
2: -0.3259546, 0.5277624, -0.1979575, 0.1493119, -0.4752666, 0.7257199
3: -0.6094154, 0.6039184, -0.2790203, 0.1848386, -0.7942539, 0.8829387
4: -0.3659086, 0.6728157, -0.1823387, 0.1798168, -0.5457255, 0.8551544

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2652147, 0.3477954, -0.1756259, 0.1411821, -0.4063968, 0.5234213
1: -0.4077931, 0.4683562, -0.2377549, 0.2046857, -0.6124789, 0.7061111
2: -0.3259546, 0.5277624, -0.2134545, 0.2086003, -0.5345550, 0.7412169
3: -0.6094154, 0.6039184, -0.3056127, 0.2426994, -0.8521147, 0.9095311
4: -0.3659086, 0.6728157, -0.1975847, 0.2468585, -0.6127671, 0.8704004

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2495804, upper bound: 3.2538683
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2693666, 0.3691269, -0.1640568, 0.0970304, -0.3663969, 0.5331837
1: -0.4349672, 0.5040386, -0.2160547, 0.1573067, -0.5922740, 0.7200933
2: -0.3306690, 0.5494190, -0.1979575, 0.1493119, -0.4799809, 0.7473766
3: -0.6472651, 0.6534121, -0.2790203, 0.1848386, -0.8321037, 0.9324324
4: -0.3854734, 0.6977538, -0.1823387, 0.1798168, -0.5652902, 0.8800925

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2769079, upper bound: 3.3495942
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2775668, upper bound: 3.3543547
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2775668, upper bound: 3.3543547
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2693666, 0.3691269, -0.1756259, 0.1411821, -0.4105486, 0.5447527
1: -0.4349672, 0.5040386, -0.2377549, 0.2046857, -0.6396530, 0.7417935
2: -0.3306690, 0.5494190, -0.2134545, 0.2086003, -0.5392693, 0.7628735
3: -0.6472651, 0.6534121, -0.3056127, 0.2426994, -0.8899645, 0.9590248
4: -0.3854734, 0.6977538, -0.1975847, 0.2468585, -0.6323318, 0.8953385

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2760350, upper bound: 3.3615592
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2617294, upper bound: 3.3001969
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4979653, 0.7299594, -0.1531032, 0.0723666, -0.5703319, 0.8830626
1: -0.8140457, 1.0286031, -0.2108820, 0.1159620, -0.9300076, 1.2394851
2: -0.5478433, 1.1171895, -0.1965333, 0.1012961, -0.6491395, 1.3137228
3: -1.3662049, 1.3533490, -0.2056489, 0.1457631, -1.5119679, 1.5589979
4: -0.7917522, 1.4799020, -0.1241231, 0.1295653, -0.9213175, 1.6040251

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2735754, upper bound: 3.3520217
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2731357, upper bound: 3.3488466
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3824591, 0.5339618, -0.1555604, 0.0857961, -0.4682552, 0.6895222
1: -0.6256671, 0.7438667, -0.2158905, 0.1343637, -0.7600307, 0.9597571
2: -0.4322708, 0.8152196, -0.2000806, 0.1217305, -0.5540013, 1.0153003
3: -0.9977122, 0.9795502, -0.2145249, 0.1663484, -1.1640606, 1.1940751
4: -0.5841303, 1.0738608, -0.1282436, 0.1528131, -0.7369434, 1.2021043

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2492395, upper bound: 3.2539285
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2492395, upper bound: 3.2955470
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4109044, 0.5794168, -0.1528277, 0.0781285, -0.4890329, 0.7322445
1: -0.6739521, 0.7992181, -0.2117595, 0.1148102, -0.7887623, 1.0109776
2: -0.4597468, 0.8980460, -0.1970833, 0.0998014, -0.5595481, 1.0951294
3: -1.0966667, 1.0596292, -0.2046368, 0.1431707, -1.2398374, 1.2642660
4: -0.6337426, 1.1888374, -0.1242587, 0.1205573, -0.7542998, 1.3130962

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894936, upper bound: 3.3567732
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2890434, upper bound: 3.3535416
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2881652, 0.3994725, -0.1538953, 0.0847333, -0.3728985, 0.5533677
1: -0.4703782, 0.5547249, -0.2140015, 0.1236971, -0.5940754, 0.7687265
2: -0.3435284, 0.5959076, -0.1986294, 0.1090919, -0.4526204, 0.7945370
3: -0.7070373, 0.7215918, -0.2086054, 0.1530888, -0.8601261, 0.9301972
4: -0.4266806, 0.7662221, -0.1261868, 0.1310454, -0.5577260, 0.8924090

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2755814, upper bound: 3.2987806
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2759412, upper bound: 3.2997399
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.5662124, 0.8654976, -1.0179069, 0.6538599
1: -0.2129075, 0.1379279, -0.9263089, 1.2271175, -1.4400250, 1.0642368
2: -0.1962777, 0.1301043, -0.6173120, 1.3232578, -1.5195355, 0.7474163
3: -0.2156065, 0.1684411, -1.5797174, 1.6085356, -1.8241421, 1.7481586
4: -0.1264155, 0.1653319, -0.9259796, 1.7451435, -1.8715590, 1.0913115

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3031061, upper bound: 3.2884846
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2946677, upper bound: 3.2866646
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.7688816, 1.3379567, -1.4903660, 0.8565291
1: -0.2129075, 0.1379279, -1.2615764, 1.8789687, -2.0918763, 1.3995043
2: -0.1962777, 0.1301043, -0.8392357, 1.9759603, -2.1722379, 0.9693400
3: -0.2156065, 0.1684411, -2.2099347, 2.4143124, -2.6299188, 2.3783758
4: -0.1264155, 0.1653319, -1.3032722, 2.5437152, -2.6701307, 1.4686041

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3031061, upper bound: 3.2884846
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2946677, upper bound: 3.2866646
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.5662124, 0.8654976, -1.0202211, 0.6410798
1: -0.2135892, 0.1191348, -0.9263089, 1.2271175, -1.4407067, 1.0454432
2: -0.1987838, 0.1062041, -0.6173120, 1.3232578, -1.5220416, 0.7235160
3: -0.2093757, 0.1484058, -1.5797174, 1.6085356, -1.8179114, 1.7281232
4: -0.1251773, 0.1360584, -0.9259796, 1.7451435, -1.8703208, 1.0620379

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.7688816, 1.3379567, -1.4926802, 0.8437490
1: -0.2135892, 0.1191348, -1.2615764, 1.8789687, -2.0925579, 1.3807111
2: -0.1987838, 0.1062041, -0.8392357, 1.9759603, -2.1747441, 0.9454398
3: -0.2093757, 0.1484058, -2.2099347, 2.4143124, -2.6236880, 2.3583403
4: -0.1251773, 0.1360584, -1.3032722, 2.5437152, -2.6688926, 1.4393307

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.5049574, 0.7416355, -0.8940449, 0.5926049
1: -0.2129075, 0.1379279, -0.8283286, 1.0456173, -1.2585248, 0.9662565
2: -0.1962777, 0.1301043, -0.5578265, 1.1398011, -1.3360789, 0.6879308
3: -0.2156065, 0.1684411, -1.3909674, 1.3781103, -1.5937167, 1.5594084
4: -0.1264155, 0.1653319, -0.8091390, 1.5109537, -1.6373692, 0.9744709

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3599255, upper bound: 3.3041514
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2992607, upper bound: 3.2884382
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1524094, 0.0876475, -0.5825652, 0.9294757, -1.0818851, 0.6702127
1: -0.2129075, 0.1379279, -0.9631144, 1.3045405, -1.5174479, 1.1010423
2: -0.1962777, 0.1301043, -0.6453928, 1.4036360, -1.5999137, 0.7754970
3: -0.2156065, 0.1684411, -1.6579331, 1.6985271, -1.9141335, 1.8263743
4: -0.1264155, 0.1653319, -0.9648626, 1.8469262, -1.9733417, 1.1301943

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3599255, upper bound: 3.3041514
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2992607, upper bound: 3.2884382
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.5049574, 0.7416355, -0.8963590, 0.5798248
1: -0.2135892, 0.1191348, -0.8283286, 1.0456173, -1.2592065, 0.9474633
2: -0.1987838, 0.1062041, -0.5578265, 1.1398011, -1.3385849, 0.6640306
3: -0.2093757, 0.1484058, -1.3909674, 1.3781103, -1.5874860, 1.5393732
4: -0.1251773, 0.1360584, -0.8091390, 1.5109537, -1.6361309, 0.9451974

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1547235, 0.0748674, -0.5825652, 0.9294757, -1.0841992, 0.6574326
1: -0.2135892, 0.1191348, -0.9631144, 1.3045405, -1.5181297, 1.0822488
2: -0.1987838, 0.1062041, -0.6453928, 1.4036360, -1.6024197, 0.7515969
3: -0.2093757, 0.1484058, -1.6579331, 1.6985271, -1.9079028, 1.8063389
4: -0.1251773, 0.1360584, -0.9648626, 1.8469262, -1.9721035, 1.1009208

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.4499060, 0.6304163, -0.8026520, 0.5760824
1: -0.2309254, 0.1855326, -0.7394965, 0.8879167, -1.1188420, 0.9250289
2: -0.2092608, 0.1860420, -0.4973597, 0.9670970, -1.1763579, 0.6834016
3: -0.2956546, 0.2189267, -1.2005224, 1.1844325, -1.4800872, 1.4194492
4: -0.1927452, 0.2179434, -0.7111874, 1.2950562, -1.4878014, 0.9291308

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3051059, upper bound: 3.2956907
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2966645, upper bound: 3.2938166
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1722357, 0.1261764, -0.6101388, 0.9512964, -1.1235321, 0.7363151
1: -0.2309254, 0.1855326, -1.0195285, 1.3569877, -1.5879130, 1.2050611
2: -0.2092608, 0.1860420, -0.6776520, 1.4618651, -1.6711259, 0.8636939
3: -0.2956546, 0.2189267, -1.7547441, 1.7836735, -2.0793281, 1.9736700
4: -0.1927452, 0.2179434, -1.0314257, 1.9252907, -2.1180358, 1.2493690

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3051059, upper bound: 3.2956907
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2966645, upper bound: 3.2938166
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1518528, 0.0670647, -0.5721087, 0.8602238, -1.0120766, 0.6391734
1: -0.2094918, 0.0995202, -0.9281335, 1.2222875, -1.4317793, 1.0276535
2: -0.1955508, 0.0855860, -0.6169652, 1.3261545, -1.5217053, 0.7025512
3: -0.2001026, 0.1254789, -1.6040950, 1.6038598, -1.8039623, 1.7295738
4: -0.1212632, 0.1056018, -0.9273438, 1.7542982, -1.8755615, 1.0329456

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2852479
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1528528, 0.0732328, -0.3660918, 0.5122857, -0.6651385, 0.4393246
1: -0.2114542, 0.1079726, -0.5990825, 0.7203152, -0.9317694, 0.7070550
2: -0.1970211, 0.0937305, -0.4136709, 0.7746083, -0.9716294, 0.5074015
3: -0.2032855, 0.1348763, -0.9417316, 0.9491390, -1.1524246, 1.0766077
4: -0.1230521, 0.1147754, -0.5631769, 1.0179145, -1.1409667, 0.6779521

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2900250, upper bound: 3.2795784
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2938095, upper bound: 3.2833261
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2938095, upper bound: 3.2833261
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1686913, 0.1113859, -0.4540939, 0.6571444, -0.8258357, 0.5654798
1: -0.2244042, 0.1675183, -0.7423609, 0.9204669, -1.1448711, 0.9098791
2: -0.2047532, 0.1631480, -0.5018728, 0.9998336, -1.2045869, 0.6650208
3: -0.2853875, 0.1974681, -1.2258428, 1.2148058, -1.5001934, 1.4233108
4: -0.1882169, 0.1906350, -0.7120590, 1.3295162, -1.5177331, 0.9026940

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.3003830
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.3007759
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1518528, 0.0670647, -0.4540939, 0.6571444, -0.8089973, 0.5211586
1: -0.2094918, 0.0995202, -0.7423609, 0.9204669, -1.1299586, 0.8418810
2: -0.1955508, 0.0855860, -0.5018728, 0.9998336, -1.1953844, 0.5874588
3: -0.2001026, 0.1254789, -1.2258428, 1.2148058, -1.4149084, 1.3513217
4: -0.1212632, 0.1056018, -0.7120590, 1.3295162, -1.4507794, 0.8176609

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2860204
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3022634, upper bound: 3.2860204
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1791718, 0.1613497, -0.2386905, 0.2681638, -0.4473356, 0.4000402
1: -0.2454292, 0.2264803, -0.3378343, 0.3662193, -0.6116486, 0.5643146
2: -0.2181719, 0.2404808, -0.2909162, 0.3992847, -0.6174566, 0.5313970
3: -0.3187436, 0.2694265, -0.4589484, 0.4551277, -0.7738713, 0.7283749
4: -0.2011959, 0.2918157, -0.2859897, 0.4943437, -0.6955396, 0.5778054

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2567721
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2553229
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1791718, 0.1613497, -0.2477777, 0.3165118, -0.4956836, 0.4091274
1: -0.2454292, 0.2264803, -0.3687550, 0.4244840, -0.6699132, 0.5952353
2: -0.2181719, 0.2404808, -0.3046838, 0.4708307, -0.6890026, 0.5451646
3: -0.3187436, 0.2694265, -0.5380337, 0.5408742, -0.8596178, 0.8074602
4: -0.2011959, 0.2918157, -0.3157976, 0.5883580, -0.7895539, 0.6076133

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2753761
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2418018, upper bound: 3.2753954
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3763297, 0.5412447, -0.4437144, 0.6394674, -1.0157969, 0.9849591
1: -0.6118441, 0.7479995, -0.7238941, 0.8820208, -1.4938648, 1.4718935
2: -0.4276588, 0.8197721, -0.4936275, 0.9763694, -1.4040282, 1.3133997
3: -0.9750521, 0.9866915, -1.1795777, 1.1707634, -2.1458154, 2.1662693
4: -0.5933742, 1.0839796, -0.7038773, 1.3035572, -1.8969314, 1.7878568

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2725631
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2725631
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.5307240, 0.7727786, -0.4437144, 0.6394674, -1.1701910, 1.2164930
1: -0.8595346, 1.0570936, -0.7238941, 0.8820208, -1.7415555, 1.7809877
2: -0.5739172, 1.1971703, -0.4936275, 0.9763694, -1.5502867, 1.6907978
3: -1.4676929, 1.3997686, -1.1795777, 1.1707634, -2.6384561, 2.5793455
4: -0.8419453, 1.6014760, -0.7038773, 1.3035572, -2.1455026, 2.3053527

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2861896
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2524335, upper bound: 3.2861896
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5079609, 0.7406304, -0.4065894, 0.5850965, -1.0930574, 1.1472195
1: -0.8270459, 1.0331752, -0.6642164, 0.8050436, -1.6320894, 1.6973916
2: -0.5495639, 1.1436493, -0.4585306, 0.8880199, -1.4375839, 1.6021799
3: -1.4009477, 1.3651351, -1.0665126, 1.0669708, -2.4679179, 2.4316478
4: -0.8036790, 1.5224344, -0.6432256, 1.1817484, -1.9854270, 2.1656599

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3512798
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3512972
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5079609, 0.7406304, -0.6012644, 0.9550356, -1.4629962, 1.3418945
1: -0.8270459, 1.0331752, -0.9971008, 1.3104172, -2.1374631, 2.0302761
2: -0.5495639, 1.1436493, -0.6748735, 1.4533428, -2.0029068, 1.8185229
3: -1.4009477, 1.3651351, -1.7128837, 1.7140588, -3.1150060, 3.0780187
4: -0.8036790, 1.5224344, -1.0037088, 1.9098605, -2.7135394, 2.5261431

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3512798
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727785, upper bound: 3.3512972
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.4874771, 0.7130252, -0.5761735, 0.8669006, -1.3543776, 1.2891986
1: -0.8010784, 1.0116723, -0.9307029, 1.1916245, -1.9927030, 1.9423752
2: -0.5397633, 1.0839124, -0.6188431, 1.3351293, -1.8748926, 1.7027555
3: -1.3093535, 1.3407466, -1.6083661, 1.5656216, -2.8749750, 2.9491129
4: -0.7897369, 1.4399812, -0.9195610, 1.7740270, -2.5637639, 2.3595421

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2423825, upper bound: 3.2574840
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2423825, upper bound: 3.2618185
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.4548401, 0.6595957, -0.4379585, 0.6136914, -1.0685314, 1.0975540
1: -0.7501289, 0.9414754, -0.7125669, 0.8403889, -1.5905174, 1.6540422
2: -0.5089628, 0.9987832, -0.4827589, 0.9517658, -1.4607286, 1.4815421
3: -1.2180669, 1.2449627, -1.1754504, 1.1185763, -2.3366432, 2.4204125
4: -0.7325845, 1.3300655, -0.6832772, 1.2810770, -2.0136614, 2.0133426

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2422209, upper bound: 3.2571351
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2422209, upper bound: 3.3458293
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.5181233, 0.7545642, -0.5761735, 0.8669006, -1.3850238, 1.3307374
1: -0.8484849, 1.0746478, -0.9307029, 1.1916245, -2.0401094, 2.0053506
2: -0.5671877, 1.1613058, -0.6188431, 1.3351293, -1.9023169, 1.7801489
3: -1.4194343, 1.4219347, -1.6083661, 1.5656216, -2.9850559, 3.0303009
4: -0.8377336, 1.5415409, -0.9195610, 1.7740270, -2.6117606, 2.4611018

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2477446, upper bound: 3.2754979
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2477446, upper bound: 3.3300085
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.4482613, 0.6359648, -0.4379585, 0.6136914, -1.0619525, 1.0739232
1: -0.7355444, 0.8976643, -0.7125669, 0.8403889, -1.5759331, 1.6102312
2: -0.4994430, 0.9721082, -0.4827589, 0.9517658, -1.4512086, 1.4548669
3: -1.1984807, 1.1885139, -1.1754504, 1.1185763, -2.3170571, 2.3639641
4: -0.7057335, 1.2945760, -0.6832772, 1.2810770, -1.9868106, 1.9778533

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2455087, upper bound: 3.2669055
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2455087, upper bound: 3.3445908
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.7774747, 1.3240056, -0.6716200, 1.0484600, -1.8259346, 1.9956255
1: -1.2421060, 1.8495609, -1.0925409, 1.4913861, -2.7334919, 2.9421017
2: -0.8266892, 1.9650905, -0.7283928, 1.5999331, -2.4266224, 2.6934829
3: -2.2023203, 2.3728528, -1.9068654, 1.9453235, -4.1476436, 4.2797184
4: -1.2790440, 2.5647111, -1.1054642, 2.1285594, -3.4076033, 3.6701753

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3138073, upper bound: 3.3597445
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205317, upper bound: 3.3619826
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.7774747, 1.3240056, -0.4382297, 0.6320060, -1.4094807, 1.7622353
1: -1.2421060, 1.8495609, -0.7168468, 0.8857412, -2.1278472, 2.5664077
2: -0.8266892, 1.9650905, -0.4834656, 0.9657283, -1.7924172, 2.4485557
3: -2.2023203, 2.3728528, -1.1678896, 1.1731887, -3.3755090, 3.5407424
4: -1.2790440, 2.5647111, -0.6880599, 1.2841494, -2.5631933, 3.2527709

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3138073, upper bound: 3.3597445
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205317, upper bound: 3.3619826
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.6623453, 1.0995708, -0.5583415, 0.8277088, -1.4900540, 1.6579123
1: -1.0661137, 1.5550854, -0.9042768, 1.1698270, -2.2359402, 2.4593616
2: -0.7131914, 1.6410513, -0.5974064, 1.2688980, -1.9820894, 2.2384577
3: -1.8519288, 2.0045834, -1.5432237, 1.5357510, -3.3876798, 3.5478067
4: -1.0997559, 2.1411972, -0.8885157, 1.6892197, -2.7889757, 3.0297129

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.8224408, 1.4189109, -0.5583415, 0.8277088, -1.6501493, 1.9772522
1: -1.3124753, 1.9887803, -0.9042768, 1.1698270, -2.4823012, 2.8930564
2: -0.8744270, 2.0936487, -0.5974064, 1.2688980, -2.1433251, 2.6910551
3: -2.3297238, 2.5440319, -1.5432237, 1.5357510, -3.8654747, 4.0872550
4: -1.3579844, 2.7262599, -0.8885157, 1.6892197, -3.0472040, 3.6147757

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3275613, upper bound: 3.3707629
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3245428, 0.4519358, -0.4084996, 0.5693522, -0.8938949, 0.8604354
1: -0.5311444, 0.6332443, -0.6703718, 0.7839134, -1.3150578, 1.3036158
2: -0.3751126, 0.6765136, -0.4531251, 0.8762918, -1.2514043, 1.1296387
3: -0.8123156, 0.8308139, -1.0841039, 1.0413346, -1.8536496, 1.9149175
4: -0.4943081, 0.8798682, -0.6257645, 1.1637021, -1.6580101, 1.5056326

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3091331
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3091331
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3091331
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3245428, 0.4519358, -0.3069124, 0.4287994, -0.7533420, 0.7588482
1: -0.5311444, 0.6332443, -0.5068933, 0.5922535, -1.1233976, 1.1401377
2: -0.3751126, 0.6765136, -0.3592525, 0.6426993, -1.0178119, 1.0357661
3: -0.8123156, 0.8308139, -0.7735788, 0.7753519, -1.5876675, 1.6043926
4: -0.4943081, 0.8798682, -0.4579649, 0.8279979, -1.3223060, 1.3378330

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3125765
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3126019
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3008626, upper bound: 3.3126019
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2881652, 0.3994725, -0.4084996, 0.5693522, -0.8575174, 0.8079721
1: -0.4703782, 0.5547249, -0.6703718, 0.7839134, -1.2542915, 1.2250966
2: -0.3435284, 0.5959076, -0.4531251, 0.8762918, -1.2198203, 1.0490328
3: -0.7070373, 0.7215918, -1.0841039, 1.0413346, -1.7483716, 1.8056958
4: -0.4266806, 0.7662221, -0.6257645, 1.1637021, -1.5903825, 1.3919866

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3029562
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2881652, 0.3994725, -0.3069124, 0.4287994, -0.7169645, 0.7063849
1: -0.4703782, 0.5547249, -0.5068933, 0.5922535, -1.0626315, 1.0616183
2: -0.3435284, 0.5959076, -0.3592525, 0.6426993, -0.9862278, 0.9551601
3: -0.7070373, 0.7215918, -0.7735788, 0.7753519, -1.4823890, 1.4951706
4: -0.4266806, 0.7662221, -0.4579649, 0.8279979, -1.2546785, 1.2241869

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3081054
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994083, upper bound: 3.3081054
time: 0.33 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.3843838, 0.5442983, -1.3908604, 1.8510292
1: -1.3568091, 2.0639806, -0.6261952, 0.7845780, -2.1413870, 2.6901758
2: -0.9353622, 2.1358516, -0.4250736, 0.8133317, -1.7486939, 2.5609252
3: -2.3087072, 2.6659245, -0.9829251, 1.0492945, -3.3580017, 3.6488495
4: -1.4961721, 2.7389774, -0.6316682, 1.0868547, -2.5830269, 3.3706448

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.5824219, 0.8309333, -1.6774955, 2.0490673
1: -1.3568091, 2.0639806, -0.9489889, 1.2061052, -2.5629134, 3.0129690
2: -0.9353622, 2.1358516, -0.6298509, 1.2675221, -2.2028842, 2.7657018
3: -2.3087072, 2.6659245, -1.5705235, 1.6317157, -3.9404230, 4.2364469
4: -1.4961721, 2.7389774, -0.9909920, 1.7309525, -3.2271247, 3.7299690

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.4113527, 0.5663913, -1.4129535, 1.8779980
1: -1.3568091, 2.0639806, -0.6659899, 0.8390931, -2.1959014, 2.7299705
2: -0.9353622, 2.1358516, -0.4494389, 0.8429364, -1.7782986, 2.5852904
3: -2.3087072, 2.6659245, -1.0422056, 1.1209043, -3.4296112, 3.7081301
4: -1.4961721, 2.7389774, -0.6735705, 1.1254845, -2.6216564, 3.4125471

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.6389702, 0.8961046, -1.7426667, 2.1056156
1: -1.3568091, 2.0639806, -1.0339110, 1.3050261, -2.6618345, 3.0978916
2: -0.9353622, 2.1358516, -0.6806795, 1.3768083, -2.3121700, 2.8165312
3: -2.3087072, 2.6659245, -1.7283909, 1.7692897, -4.0779963, 4.3943152
4: -1.4961721, 2.7389774, -1.0785359, 1.8915758, -3.3877475, 3.8175120

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_A2_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.3843838, 0.5442983, -1.4130764, 1.8703843
1: -1.3866758, 2.1095767, -0.6261952, 0.7845780, -2.1712537, 2.7357719
2: -0.9571933, 2.1844273, -0.4250736, 0.8133317, -1.7705250, 2.6095009
3: -2.3729274, 2.7349393, -0.9829251, 1.0492945, -3.4222219, 3.7178645
4: -1.5417966, 2.8029537, -0.6316682, 1.0868547, -2.6286511, 3.4346218

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.5824219, 0.8309333, -1.6997112, 2.0684223
1: -1.3866758, 2.1095767, -0.9489889, 1.2061052, -2.5927808, 3.0585651
2: -0.9571933, 2.1844273, -0.6298509, 1.2675221, -2.2247152, 2.8142776
3: -2.3729274, 2.7349393, -1.5705235, 1.6317157, -4.0046430, 4.3054628
4: -1.5417966, 2.8029537, -0.9909920, 1.7309525, -3.2727489, 3.7939458

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_A2_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.4117772, 0.5697080, -1.4384860, 1.8977778
1: -1.3866758, 2.1095767, -0.6674201, 0.8443651, -2.2310410, 2.7769969
2: -0.9571933, 2.1844273, -0.4508835, 0.8464034, -1.8035967, 2.6353106
3: -2.3729274, 2.7349393, -1.0447898, 1.1282214, -3.5011487, 3.7797291
4: -1.5417966, 2.8029537, -0.6764904, 1.1303128, -2.6721094, 3.4794440

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.6491946, 0.9202846, -1.7890624, 2.1351953
1: -1.3866758, 2.1095767, -1.0591128, 1.3438410, -2.7305167, 3.1686893
2: -0.9571933, 2.1844273, -0.6981657, 1.4111581, -2.3683510, 2.8825929
3: -2.3729274, 2.7349393, -1.7673838, 1.8260942, -4.1990213, 4.5023232
4: -1.5417966, 2.8029537, -1.1076336, 1.9449443, -3.4867401, 3.9105873

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8364059, 1.4454074, -0.4714766, 0.6625065, -1.4989123, 1.9168839
1: -1.3392069, 2.0378947, -0.7764031, 0.9376959, -2.2769027, 2.8142979
2: -0.9222740, 2.1100194, -0.5153374, 1.0052826, -1.9275566, 2.6253569
3: -2.2800412, 2.6337733, -1.2548405, 1.2705309, -3.5505722, 3.8886137
4: -1.4792006, 2.6968720, -0.7883511, 1.3532872, -2.8324871, 3.4852231

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8937695, upper bound: 2.9017946
time: 0.32 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8937695, upper bound: 3.0736356
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8364059, 1.4454074, -0.7411959, 1.0899432, -1.9263489, 2.1866031
1: -1.3392069, 2.0378947, -1.2066995, 1.5160031, -2.8552101, 3.2445936
2: -0.9222740, 2.1100194, -0.7900785, 1.7891679, -2.7114418, 2.9000978
3: -2.2800412, 2.6337733, -2.1132293, 2.0466590, -4.3266997, 4.7470026
4: -1.4792006, 2.6968720, -1.2572596, 2.3491647, -3.8283644, 3.9541306

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8937695, upper bound: 2.9108475
time: 0.35 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8937695, upper bound: 3.0736356
time: 0.35 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.8633735, 1.4755141, -0.4714766, 0.6625065, -1.5258801, 1.9469906
1: -1.3774875, 2.0958517, -0.7764031, 0.9376959, -2.3151832, 2.8722548
2: -0.9503216, 2.1727619, -0.5153374, 1.0052826, -1.9556042, 2.6880994
3: -2.3591931, 2.7177744, -1.2548405, 1.2705309, -3.6297240, 3.9726148
4: -1.5330763, 2.7828827, -0.7883511, 1.3532872, -2.8863630, 3.5712337

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8633735, 1.4755141, -0.7411959, 1.0899432, -1.9533167, 2.2167091
1: -1.3774875, 2.0958517, -1.2066995, 1.5160031, -2.8934906, 3.3025506
2: -0.9503216, 2.1727619, -0.7900785, 1.7891679, -2.7394896, 2.9628403
3: -2.3591931, 2.7177744, -2.1132293, 2.0466590, -4.4058523, 4.8310037
4: -1.5330763, 2.7828827, -1.2572596, 2.3491647, -3.8822410, 4.0401416

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -0.6942571, 1.1081522, -1.9547143, 2.1609025
1: -1.3568091, 2.0639806, -1.1206880, 1.5614812, -2.9182899, 3.1846685
2: -0.9353622, 2.1358516, -0.7438019, 1.6577936, -2.5931559, 2.8796535
3: -2.3087072, 2.6659245, -1.9121609, 2.0466893, -4.3553963, 4.5780845
4: -1.4961721, 2.7389774, -1.1777240, 2.1780043, -3.6741762, 3.9167013

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8465621, 1.4666455, -1.0964229, 1.9137828, -2.7603447, 2.5630682
1: -1.3568091, 2.0639806, -1.7346299, 2.6876557, -4.0444651, 3.7986104
2: -0.9353622, 2.1358516, -1.1982838, 2.8121591, -3.7475209, 3.3341353
3: -2.3087072, 2.6659245, -3.0174291, 3.4654942, -5.7742004, 5.6833534
4: -1.4961721, 2.7389774, -1.9191794, 3.6119740, -5.1081462, 4.6581569

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -0.6942571, 1.1081522, -1.9769304, 2.1802578
1: -1.3866758, 2.1095767, -1.1206880, 1.5614812, -2.9481571, 3.2302647
2: -0.9571933, 2.1844273, -0.7438019, 1.6577936, -2.6149869, 2.9282291
3: -2.3729274, 2.7349393, -1.9121609, 2.0466893, -4.4196167, 4.6471004
4: -1.5417966, 2.8029537, -1.1777240, 2.1780043, -3.7198009, 3.9806776

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.8687782, 1.4860005, -1.0964229, 1.9137828, -2.7825611, 2.5824234
1: -1.3866758, 2.1095767, -1.7346299, 2.6876557, -4.0743308, 3.8442066
2: -0.9571933, 2.1844273, -1.1982838, 2.8121591, -3.7693524, 3.3827109
3: -2.3729274, 2.7349393, -3.0174291, 3.4654942, -5.8384209, 5.7523685
4: -1.5417966, 2.8029537, -1.9191794, 3.6119740, -5.1537704, 4.7221332

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.92 + 306.32 = 308.24 seconds
