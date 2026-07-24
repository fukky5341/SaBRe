## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.012884755


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974)
1: (-0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598)
2: (-0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447)
3: (-0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147)
4: (-0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 0.75 = 1.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0135629, upper bound: 0.0135629

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131984, upper bound: 0.0131819
time: 0.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
time: 0.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.45 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0131984, upper bound: 0.0131819
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0139534, 0.0011155, -0.0142281, 0.0014693, -0.0154227, 0.0153436
1: -0.0315158, 0.0094275, -0.0323874, 0.0100724, -0.0415882, 0.0418149
2: -0.0290593, 0.0157219, -0.0295564, 0.0169883, -0.0460475, 0.0452783
3: -0.0306743, 0.0147229, -0.0313897, 0.0155250, -0.0461993, 0.0461126
4: -0.0248440, 0.0179260, -0.0253857, 0.0193347, -0.0441787, 0.0433117

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
time: 0.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
time: 0.19 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0147513, 0.0034798, -0.0142281, 0.0014693, -0.0162206, 0.0177080
1: -0.0360003, 0.0115828, -0.0323874, 0.0100724, -0.0460727, 0.0439702
2: -0.0292871, 0.0257617, -0.0295564, 0.0169883, -0.0462753, 0.0553181
3: -0.0336355, 0.0188496, -0.0313897, 0.0155250, -0.0491605, 0.0502393
4: -0.0269840, 0.0281143, -0.0253857, 0.0193347, -0.0463187, 0.0535000

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
time: 0.19 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
time: 0.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.20 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0129311, upper bound: 0.0129311

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0139534, 0.0011155, -0.0139534, 0.0011155, -0.0150689, 0.0150689
1: -0.0315158, 0.0094275, -0.0315158, 0.0094275, -0.0409433, 0.0409433
2: -0.0290593, 0.0157219, -0.0290593, 0.0157219, -0.0447812, 0.0447812
3: -0.0306743, 0.0147229, -0.0306743, 0.0147229, -0.0453972, 0.0453972
4: -0.0248440, 0.0179260, -0.0248440, 0.0179260, -0.0427700, 0.0427700

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131496, upper bound: 0.0114366
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131932, upper bound: 0.0131710
time: 0.22 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0139534, 0.0011155, -0.0147513, 0.0034798, -0.0174332, 0.0158668
1: -0.0315158, 0.0094275, -0.0360003, 0.0115828, -0.0430985, 0.0454278
2: -0.0290593, 0.0157219, -0.0292871, 0.0257617, -0.0548210, 0.0450090
3: -0.0306743, 0.0147229, -0.0336355, 0.0188496, -0.0495239, 0.0483584
4: -0.0248440, 0.0179260, -0.0269840, 0.0281143, -0.0529583, 0.0449100

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131496, upper bound: 0.0114366
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131932, upper bound: 0.0131710
time: 0.20 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0147513, 0.0034798, -0.0139534, 0.0011155, -0.0158668, 0.0174332
1: -0.0360003, 0.0115828, -0.0315158, 0.0094275, -0.0454278, 0.0430985
2: -0.0292871, 0.0257617, -0.0290593, 0.0157219, -0.0450090, 0.0548210
3: -0.0336355, 0.0188496, -0.0306743, 0.0147229, -0.0483584, 0.0495239
4: -0.0269840, 0.0281143, -0.0248440, 0.0179260, -0.0449100, 0.0529583

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0129150
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127961, upper bound: 0.0127961
time: 0.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0147513, 0.0034798, -0.0147513, 0.0034798, -0.0182311, 0.0182311
1: -0.0360003, 0.0115828, -0.0360003, 0.0115828, -0.0475831, 0.0475831
2: -0.0292871, 0.0257617, -0.0292871, 0.0257617, -0.0550488, 0.0550488
3: -0.0336355, 0.0188496, -0.0336355, 0.0188496, -0.0524851, 0.0524851
4: -0.0269840, 0.0281143, -0.0269840, 0.0281143, -0.0550983, 0.0550983

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0129150
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127961, upper bound: 0.0127961
time: 0.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.40 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0131496, upper bound: 0.0114366
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0131932, upper bound: 0.0131710
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0131496, upper bound: 0.0114366
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0131932, upper bound: 0.0131710
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0129150
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0127961, upper bound: 0.0127961
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0129150
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0127961, upper bound: 0.0127961

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0117640, -0.0037626, -0.0139534, 0.0011155, -0.0128795, 0.0101908
1: -0.0202981, 0.0001338, -0.0315158, 0.0094275, -0.0297256, 0.0316496
2: -0.0243619, -0.0031668, -0.0290593, 0.0157219, -0.0400838, 0.0258925
3: -0.0192202, 0.0025270, -0.0306743, 0.0147229, -0.0339431, 0.0332012
4: -0.0202474, -0.0023814, -0.0248440, 0.0179260, -0.0381734, 0.0224626

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117346, upper bound: 0.0117346
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117346, upper bound: 0.0117648
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0139534, 0.0011155, -0.0149992, 0.0149627
1: -0.0311912, 0.0091927, -0.0315158, 0.0094275, -0.0406187, 0.0407085
2: -0.0289316, 0.0154323, -0.0290593, 0.0157219, -0.0446535, 0.0444915
3: -0.0303655, 0.0143663, -0.0306743, 0.0147229, -0.0450884, 0.0450406
4: -0.0247009, 0.0175823, -0.0248440, 0.0179260, -0.0426269, 0.0424262

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117647, upper bound: 0.0135109
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117647, upper bound: 0.0135410
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0117640, -0.0037626, -0.0147513, 0.0034798, -0.0152438, 0.0109887
1: -0.0202981, 0.0001338, -0.0360003, 0.0115828, -0.0318809, 0.0361342
2: -0.0243619, -0.0031668, -0.0292871, 0.0257617, -0.0501236, 0.0261203
3: -0.0192202, 0.0025270, -0.0336355, 0.0188496, -0.0380698, 0.0361625
4: -0.0202474, -0.0023814, -0.0269840, 0.0281143, -0.0483617, 0.0246026

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131355, upper bound: 0.0112259
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130019, upper bound: 0.0114194
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0147513, 0.0034798, -0.0173635, 0.0157606
1: -0.0311912, 0.0091927, -0.0360003, 0.0115828, -0.0427739, 0.0451931
2: -0.0289316, 0.0154323, -0.0292871, 0.0257617, -0.0546933, 0.0447193
3: -0.0303655, 0.0143663, -0.0336355, 0.0188496, -0.0492151, 0.0480018
4: -0.0247009, 0.0175823, -0.0269840, 0.0281143, -0.0528152, 0.0445662

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131791, upper bound: 0.0128749
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130489, upper bound: 0.0131445
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0145323, 0.0033897, -0.0139534, 0.0011155, -0.0156478, 0.0173431
1: -0.0353562, 0.0103078, -0.0315158, 0.0094275, -0.0447837, 0.0418236
2: -0.0282667, 0.0249883, -0.0290593, 0.0157219, -0.0439886, 0.0540475
3: -0.0329278, 0.0178143, -0.0306743, 0.0147229, -0.0476506, 0.0484886
4: -0.0261725, 0.0274723, -0.0248440, 0.0179260, -0.0440985, 0.0523163

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0131177
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128451, upper bound: 0.0131534
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0145323, 0.0033897, -0.0147513, 0.0034798, -0.0180121, 0.0181409
1: -0.0353562, 0.0103078, -0.0360003, 0.0115828, -0.0469390, 0.0463081
2: -0.0282667, 0.0249883, -0.0292871, 0.0257617, -0.0540284, 0.0542753
3: -0.0329278, 0.0178143, -0.0336355, 0.0188496, -0.0517773, 0.0514498
4: -0.0261725, 0.0274723, -0.0269840, 0.0281143, -0.0542868, 0.0544563

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0125150
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0127961
time: 0.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.26 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0117346, upper bound: 0.0117346
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0117346, upper bound: 0.0117648
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0117647, upper bound: 0.0135109
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0117647, upper bound: 0.0135410
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0131355, upper bound: 0.0112259
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0130019, upper bound: 0.0114194
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0131791, upper bound: 0.0128749
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0130489, upper bound: 0.0131445
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0131177
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0128451, upper bound: 0.0131534
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0125150
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.26
Output dim: 0, lower bound: -0.0125057, upper bound: 0.0127961

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0117640, -0.0037626, -0.0101211, 0.0127733
1: -0.0311912, 0.0091927, -0.0202981, 0.0001338, -0.0313250, 0.0294908
2: -0.0289316, 0.0154323, -0.0243619, -0.0031668, -0.0257648, 0.0397942
3: -0.0303655, 0.0143663, -0.0192202, 0.0025270, -0.0328925, 0.0335865
4: -0.0247009, 0.0175823, -0.0202474, -0.0023814, -0.0223195, 0.0378297

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117618, upper bound: 0.0133345
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117509, upper bound: 0.0133332
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0138837, 0.0010093, -0.0148930, 0.0148930
1: -0.0311912, 0.0091927, -0.0311912, 0.0091927, -0.0403839, 0.0403839
2: -0.0289316, 0.0154323, -0.0289316, 0.0154323, -0.0443638, 0.0443638
3: -0.0303655, 0.0143663, -0.0303655, 0.0143663, -0.0447319, 0.0447319
4: -0.0247009, 0.0175823, -0.0247009, 0.0175823, -0.0422831, 0.0422831

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117618, upper bound: 0.0134578
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117509, upper bound: 0.0134564
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0117640, -0.0037626, -0.0145323, 0.0033897, -0.0151537, 0.0107697
1: -0.0202981, 0.0001338, -0.0353562, 0.0103078, -0.0306059, 0.0354901
2: -0.0243619, -0.0031668, -0.0282667, 0.0249883, -0.0493502, 0.0250999
3: -0.0192202, 0.0025270, -0.0329278, 0.0178143, -0.0370345, 0.0354547
4: -0.0202474, -0.0023814, -0.0261725, 0.0274723, -0.0477197, 0.0237911

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130949, upper bound: 0.0111674
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130724, upper bound: 0.0112191
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0117640, -0.0037626, -0.0144330, 0.0031946, -0.0149586, 0.0106704
1: -0.0202981, 0.0001338, -0.0351038, 0.0101279, -0.0304260, 0.0352377
2: -0.0243619, -0.0031668, -0.0283699, 0.0250444, -0.0494063, 0.0252031
3: -0.0192202, 0.0025270, -0.0326761, 0.0174502, -0.0366704, 0.0352030
4: -0.0202474, -0.0023814, -0.0261594, 0.0273368, -0.0475842, 0.0237781

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129813, upper bound: 0.0114151
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129720, upper bound: 0.0114194
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0145323, 0.0033897, -0.0172733, 0.0155416
1: -0.0311912, 0.0091927, -0.0353562, 0.0103078, -0.0414990, 0.0445489
2: -0.0289316, 0.0154323, -0.0282667, 0.0249883, -0.0539198, 0.0436990
3: -0.0303655, 0.0143663, -0.0329278, 0.0178143, -0.0481798, 0.0472941
4: -0.0247009, 0.0175823, -0.0261725, 0.0274723, -0.0521732, 0.0437547

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131133, upper bound: 0.0124649
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131483, upper bound: 0.0128362
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138837, 0.0010093, -0.0144330, 0.0031946, -0.0170783, 0.0154423
1: -0.0311912, 0.0091927, -0.0351038, 0.0101279, -0.0413190, 0.0442965
2: -0.0289316, 0.0154323, -0.0283699, 0.0250444, -0.0539759, 0.0438021
3: -0.0303655, 0.0143663, -0.0326761, 0.0174502, -0.0478157, 0.0470424
4: -0.0247009, 0.0175823, -0.0261594, 0.0273368, -0.0520377, 0.0437417

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129881, upper bound: 0.0128115
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130414, upper bound: 0.0130875
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145323, 0.0033897, -0.0130612, 0.0002880, -0.0148203, 0.0164508
1: -0.0353562, 0.0103078, -0.0304714, 0.0083756, -0.0437318, 0.0407792
2: -0.0282667, 0.0249883, -0.0281704, 0.0039606, -0.0322273, 0.0531586
3: -0.0329278, 0.0178143, -0.0301511, 0.0135370, -0.0464648, 0.0479654
4: -0.0261725, 0.0274723, -0.0231530, 0.0062372, -0.0324097, 0.0506253

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0130871
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0130859
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145323, 0.0033897, -0.0137183, 0.0009172, -0.0154495, 0.0171080
1: -0.0353562, 0.0103078, -0.0310025, 0.0089382, -0.0442944, 0.0413103
2: -0.0282667, 0.0249883, -0.0286273, 0.0153685, -0.0436352, 0.0536156
3: -0.0329278, 0.0178143, -0.0301444, 0.0141103, -0.0470381, 0.0479587
4: -0.0261725, 0.0274723, -0.0244682, 0.0174519, -0.0436244, 0.0519405

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0131246
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125210, upper bound: 0.0131224
time: 0.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.50 seconds
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0117618, upper bound: 0.0133345
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0117509, upper bound: 0.0133332
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0117618, upper bound: 0.0134578
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0117509, upper bound: 0.0134564
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0130949, upper bound: 0.0111674
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0130724, upper bound: 0.0112191
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0129813, upper bound: 0.0114151
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0129720, upper bound: 0.0114194
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0131133, upper bound: 0.0124649
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0131483, upper bound: 0.0128362
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0129881, upper bound: 0.0128115
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0130414, upper bound: 0.0130875
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0130871
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0130859
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0131246
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -0.0125210, upper bound: 0.0131224

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0117640, -0.0037626, -0.0092408, 0.0119597
1: -0.0302214, 0.0081966, -0.0202981, 0.0001338, -0.0303553, 0.0284947
2: -0.0280575, 0.0037971, -0.0243619, -0.0031668, -0.0248908, 0.0281591
3: -0.0299382, 0.0132969, -0.0192202, 0.0025270, -0.0324652, 0.0325171
4: -0.0230227, 0.0060431, -0.0202474, -0.0023814, -0.0206413, 0.0262905

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0117640, -0.0037626, -0.0098916, 0.0125906
1: -0.0307293, 0.0087318, -0.0202981, 0.0001338, -0.0308631, 0.0290299
2: -0.0285153, 0.0151063, -0.0243619, -0.0031668, -0.0253485, 0.0394682
3: -0.0298880, 0.0137899, -0.0192202, 0.0025270, -0.0324149, 0.0330101
4: -0.0243416, 0.0171402, -0.0202474, -0.0023814, -0.0219602, 0.0373876

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0138837, 0.0010093, -0.0140127, 0.0140794
1: -0.0302214, 0.0081966, -0.0311912, 0.0091927, -0.0394141, 0.0393878
2: -0.0280575, 0.0037971, -0.0289316, 0.0154323, -0.0434898, 0.0327287
3: -0.0299382, 0.0132969, -0.0303655, 0.0143663, -0.0443045, 0.0436624
4: -0.0230227, 0.0060431, -0.0247009, 0.0175823, -0.0406049, 0.0307440

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124566, upper bound: 0.0134564
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124566, upper bound: 0.0134564
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0138837, 0.0010093, -0.0146635, 0.0147102
1: -0.0307293, 0.0087318, -0.0311912, 0.0091927, -0.0399220, 0.0399230
2: -0.0285153, 0.0151063, -0.0289316, 0.0154323, -0.0439476, 0.0440379
3: -0.0298880, 0.0137899, -0.0303655, 0.0143663, -0.0442543, 0.0441554
4: -0.0243416, 0.0171402, -0.0247009, 0.0175823, -0.0419238, 0.0418410

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126043, upper bound: 0.0134564
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126043, upper bound: 0.0134564
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0112948, -0.0041046, -0.0145323, 0.0033897, -0.0146845, 0.0104276
1: -0.0196947, -0.0005234, -0.0353562, 0.0103078, -0.0300025, 0.0348328
2: -0.0234008, -0.0066568, -0.0282667, 0.0249883, -0.0483891, 0.0216099
3: -0.0185211, 0.0020294, -0.0329278, 0.0178143, -0.0363354, 0.0349572
4: -0.0194711, -0.0049051, -0.0261725, 0.0274723, -0.0469435, 0.0212673

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0116469, -0.0039406, -0.0145323, 0.0033897, -0.0150365, 0.0105917
1: -0.0199876, -0.0001259, -0.0353562, 0.0103078, -0.0302954, 0.0352303
2: -0.0241192, -0.0032852, -0.0282667, 0.0249883, -0.0491074, 0.0249815
3: -0.0188645, 0.0021902, -0.0329278, 0.0178143, -0.0366788, 0.0351180
4: -0.0200901, -0.0025771, -0.0261725, 0.0274723, -0.0475624, 0.0235954

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0112948, -0.0041046, -0.0144330, 0.0031946, -0.0144894, 0.0103283
1: -0.0196947, -0.0005234, -0.0351038, 0.0101279, -0.0298226, 0.0345804
2: -0.0234008, -0.0066568, -0.0283699, 0.0250444, -0.0484452, 0.0217131
3: -0.0185211, 0.0020294, -0.0326761, 0.0174502, -0.0359713, 0.0347055
4: -0.0194711, -0.0049051, -0.0261594, 0.0273368, -0.0468080, 0.0212543

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0116469, -0.0039406, -0.0144330, 0.0031946, -0.0148415, 0.0104924
1: -0.0199876, -0.0001259, -0.0351038, 0.0101279, -0.0301154, 0.0349779
2: -0.0241192, -0.0032852, -0.0283699, 0.0250444, -0.0491635, 0.0250847
3: -0.0188645, 0.0021902, -0.0326761, 0.0174502, -0.0363147, 0.0348663
4: -0.0200901, -0.0025771, -0.0261594, 0.0273368, -0.0474269, 0.0235823

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0145323, 0.0033897, -0.0163930, 0.0147280
1: -0.0302214, 0.0081966, -0.0353562, 0.0103078, -0.0405292, 0.0435528
2: -0.0280575, 0.0037971, -0.0282667, 0.0249883, -0.0530458, 0.0320638
3: -0.0299382, 0.0132969, -0.0329278, 0.0178143, -0.0477525, 0.0462246
4: -0.0230227, 0.0060431, -0.0261725, 0.0274723, -0.0504950, 0.0322156

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130811, upper bound: 0.0124649
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130822, upper bound: 0.0123024
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0145323, 0.0033897, -0.0170438, 0.0153589
1: -0.0307293, 0.0087318, -0.0353562, 0.0103078, -0.0410371, 0.0440880
2: -0.0285153, 0.0151063, -0.0282667, 0.0249883, -0.0535036, 0.0433730
3: -0.0298880, 0.0137899, -0.0329278, 0.0178143, -0.0477023, 0.0467176
4: -0.0243416, 0.0171402, -0.0261725, 0.0274723, -0.0518139, 0.0433127

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131179, upper bound: 0.0128255
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131181, upper bound: 0.0125131
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0144330, 0.0031946, -0.0161980, 0.0146287
1: -0.0302214, 0.0081966, -0.0351038, 0.0101279, -0.0403493, 0.0433004
2: -0.0280575, 0.0037971, -0.0283699, 0.0250444, -0.0531019, 0.0321670
3: -0.0299382, 0.0132969, -0.0326761, 0.0174502, -0.0473884, 0.0459730
4: -0.0230227, 0.0060431, -0.0261594, 0.0273368, -0.0503595, 0.0322026

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128435, upper bound: 0.0127566
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129849, upper bound: 0.0128077
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0144330, 0.0031946, -0.0168488, 0.0152596
1: -0.0307293, 0.0087318, -0.0351038, 0.0101279, -0.0408572, 0.0438357
2: -0.0285153, 0.0151063, -0.0283699, 0.0250444, -0.0535597, 0.0434762
3: -0.0298880, 0.0137899, -0.0326761, 0.0174502, -0.0473382, 0.0464660
4: -0.0243416, 0.0171402, -0.0261594, 0.0273368, -0.0516784, 0.0432996

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129119, upper bound: 0.0130670
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130388, upper bound: 0.0130774
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0151567, 0.0043182, -0.0130612, 0.0002880, -0.0154447, 0.0173793
1: -0.0381850, 0.0127767, -0.0304714, 0.0083756, -0.0465606, 0.0432481
2: -0.0292611, 0.0297494, -0.0281704, 0.0039606, -0.0332217, 0.0579198
3: -0.0349177, 0.0209210, -0.0301511, 0.0135370, -0.0484548, 0.0510721
4: -0.0277269, 0.0324924, -0.0231530, 0.0062372, -0.0339641, 0.0556454

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0122663
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0130871
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0143511, 0.0032448, -0.0130612, 0.0002880, -0.0146391, 0.0163060
1: -0.0349527, 0.0099729, -0.0304714, 0.0083756, -0.0433282, 0.0404443
2: -0.0278904, 0.0245293, -0.0281704, 0.0039606, -0.0318510, 0.0526997
3: -0.0325822, 0.0173989, -0.0301511, 0.0135370, -0.0461192, 0.0475500
4: -0.0258082, 0.0269775, -0.0231530, 0.0062372, -0.0320453, 0.0501305

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0122637
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0130859
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0151567, 0.0043182, -0.0137183, 0.0009172, -0.0160739, 0.0180365
1: -0.0381850, 0.0127767, -0.0310025, 0.0089382, -0.0471233, 0.0437792
2: -0.0292611, 0.0297494, -0.0286273, 0.0153685, -0.0446296, 0.0583768
3: -0.0349177, 0.0209210, -0.0301444, 0.0141103, -0.0490280, 0.0510654
4: -0.0277269, 0.0324924, -0.0244682, 0.0174519, -0.0451789, 0.0569606

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0123939
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0131246
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0143511, 0.0032448, -0.0137183, 0.0009172, -0.0152683, 0.0169631
1: -0.0349527, 0.0099729, -0.0310025, 0.0089382, -0.0438909, 0.0409754
2: -0.0278904, 0.0245293, -0.0286273, 0.0153685, -0.0432589, 0.0531567
3: -0.0325822, 0.0173989, -0.0301444, 0.0141103, -0.0466925, 0.0475433
4: -0.0258082, 0.0269775, -0.0244682, 0.0174519, -0.0432601, 0.0514456

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125125, upper bound: 0.0123913
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125125, upper bound: 0.0123913
time: 0.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.30 seconds
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0117410, upper bound: 0.0133332
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0124566, upper bound: 0.0134564
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0124566, upper bound: 0.0134564
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0126043, upper bound: 0.0134564
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0126043, upper bound: 0.0134564
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0130811, upper bound: 0.0124649
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0130822, upper bound: 0.0123024
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0131179, upper bound: 0.0128255
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0131181, upper bound: 0.0125131
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0128435, upper bound: 0.0127566
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0129849, upper bound: 0.0128077
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0129119, upper bound: 0.0130670
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0130388, upper bound: 0.0130774
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0122663
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0125335, upper bound: 0.0130871
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0122637
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0123638, upper bound: 0.0130859
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0123939
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0128341, upper bound: 0.0131246
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0125125, upper bound: 0.0123913
NS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -0.0125125, upper bound: 0.0123913

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0112948, -0.0041046, -0.0088987, 0.0114905
1: -0.0302214, 0.0081966, -0.0196947, -0.0005234, -0.0296980, 0.0278913
2: -0.0280575, 0.0037971, -0.0234008, -0.0066568, -0.0214008, 0.0271980
3: -0.0299382, 0.0132969, -0.0185211, 0.0020294, -0.0319676, 0.0318180
4: -0.0230227, 0.0060431, -0.0194711, -0.0049051, -0.0181175, 0.0255143

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108209, upper bound: 0.0132342
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117404, upper bound: 0.0133315
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0116469, -0.0039406, -0.0090628, 0.0118426
1: -0.0302214, 0.0081966, -0.0199876, -0.0001259, -0.0300955, 0.0281842
2: -0.0280575, 0.0037971, -0.0241192, -0.0032852, -0.0247723, 0.0279163
3: -0.0299382, 0.0132969, -0.0188645, 0.0021902, -0.0321284, 0.0321614
4: -0.0230227, 0.0060431, -0.0200901, -0.0025771, -0.0204455, 0.0261332

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108209, upper bound: 0.0132342
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117404, upper bound: 0.0133315
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0112948, -0.0041046, -0.0095495, 0.0121214
1: -0.0307293, 0.0087318, -0.0196947, -0.0005234, -0.0302059, 0.0284265
2: -0.0285153, 0.0151063, -0.0234008, -0.0066568, -0.0218586, 0.0385071
3: -0.0298880, 0.0137899, -0.0185211, 0.0020294, -0.0319174, 0.0323110
4: -0.0243416, 0.0171402, -0.0194711, -0.0049051, -0.0194364, 0.0366113

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0111201, upper bound: 0.0132678
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117294, upper bound: 0.0133227
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0116469, -0.0039406, -0.0097136, 0.0124734
1: -0.0307293, 0.0087318, -0.0199876, -0.0001259, -0.0306034, 0.0287194
2: -0.0285153, 0.0151063, -0.0241192, -0.0032852, -0.0252301, 0.0392255
3: -0.0298880, 0.0137899, -0.0188645, 0.0021902, -0.0320782, 0.0326544
4: -0.0243416, 0.0171402, -0.0200901, -0.0025771, -0.0217644, 0.0372302

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0111201, upper bound: 0.0132678
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117294, upper bound: 0.0133227
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0130033, 0.0001957, -0.0131991, 0.0131991
1: -0.0302214, 0.0081966, -0.0302214, 0.0081966, -0.0384180, 0.0384180
2: -0.0280575, 0.0037971, -0.0280575, 0.0037971, -0.0318547, 0.0318547
3: -0.0299382, 0.0132969, -0.0299382, 0.0132969, -0.0432351, 0.0432351
4: -0.0230227, 0.0060431, -0.0230227, 0.0060431, -0.0290658, 0.0290658

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122480, upper bound: 0.0133575
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124492, upper bound: 0.0134548
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0136542, 0.0008266, -0.0138299, 0.0138499
1: -0.0302214, 0.0081966, -0.0307293, 0.0087318, -0.0389532, 0.0389259
2: -0.0280575, 0.0037971, -0.0285153, 0.0151063, -0.0431638, 0.0323125
3: -0.0299382, 0.0132969, -0.0298880, 0.0137899, -0.0437281, 0.0431848
4: -0.0230227, 0.0060431, -0.0243416, 0.0171402, -0.0401628, 0.0303847

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122480, upper bound: 0.0133575
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124492, upper bound: 0.0134548
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0130033, 0.0001957, -0.0138499, 0.0138299
1: -0.0307293, 0.0087318, -0.0302214, 0.0081966, -0.0389259, 0.0389532
2: -0.0285153, 0.0151063, -0.0280575, 0.0037971, -0.0323125, 0.0431638
3: -0.0298880, 0.0137899, -0.0299382, 0.0132969, -0.0431848, 0.0437281
4: -0.0243416, 0.0171402, -0.0230227, 0.0060431, -0.0303847, 0.0401628

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124179, upper bound: 0.0133910
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125950, upper bound: 0.0134459
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0136542, 0.0008266, -0.0144807, 0.0144807
1: -0.0307293, 0.0087318, -0.0307293, 0.0087318, -0.0394611, 0.0394611
2: -0.0285153, 0.0151063, -0.0285153, 0.0151063, -0.0436216, 0.0436216
3: -0.0298880, 0.0137899, -0.0298880, 0.0137899, -0.0436779, 0.0436779
4: -0.0243416, 0.0171402, -0.0243416, 0.0171402, -0.0414817, 0.0414817

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124179, upper bound: 0.0133910
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125950, upper bound: 0.0134459
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0151567, 0.0043182, -0.0173215, 0.0153525
1: -0.0302214, 0.0081966, -0.0381850, 0.0127767, -0.0429981, 0.0463816
2: -0.0280575, 0.0037971, -0.0292611, 0.0297494, -0.0578070, 0.0330582
3: -0.0299382, 0.0132969, -0.0349177, 0.0209210, -0.0508592, 0.0482146
4: -0.0230227, 0.0060431, -0.0277269, 0.0324924, -0.0555151, 0.0337701

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122578, upper bound: 0.0124649
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122578, upper bound: 0.0124649
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0143511, 0.0032448, -0.0162482, 0.0145469
1: -0.0302214, 0.0081966, -0.0349527, 0.0099729, -0.0401943, 0.0431492
2: -0.0280575, 0.0037971, -0.0278904, 0.0245293, -0.0525868, 0.0316875
3: -0.0299382, 0.0132969, -0.0325822, 0.0173989, -0.0473371, 0.0458791
4: -0.0230227, 0.0060431, -0.0258082, 0.0269775, -0.0500001, 0.0318513

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122583, upper bound: 0.0123024
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122583, upper bound: 0.0123024
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0151567, 0.0043182, -0.0179723, 0.0159833
1: -0.0307293, 0.0087318, -0.0381850, 0.0127767, -0.0435060, 0.0469169
2: -0.0285153, 0.0151063, -0.0292611, 0.0297494, -0.0582648, 0.0443674
3: -0.0298880, 0.0137899, -0.0349177, 0.0209210, -0.0508090, 0.0487076
4: -0.0243416, 0.0171402, -0.0277269, 0.0324924, -0.0568339, 0.0448671

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123854, upper bound: 0.0128255
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123854, upper bound: 0.0128255
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0143511, 0.0032448, -0.0168990, 0.0151777
1: -0.0307293, 0.0087318, -0.0349527, 0.0099729, -0.0407022, 0.0436845
2: -0.0285153, 0.0151063, -0.0278904, 0.0245293, -0.0530446, 0.0429966
3: -0.0298880, 0.0137899, -0.0325822, 0.0173989, -0.0472869, 0.0463721
4: -0.0243416, 0.0171402, -0.0258082, 0.0269775, -0.0513190, 0.0429483

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123860, upper bound: 0.0125001
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123860, upper bound: 0.0125131
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130033, 0.0001957, -0.0142366, 0.0030378, -0.0160411, 0.0144323
1: -0.0302214, 0.0081966, -0.0347289, 0.0097630, -0.0399844, 0.0429255
2: -0.0280575, 0.0037971, -0.0279771, 0.0244865, -0.0525440, 0.0317743
3: -0.0299382, 0.0132969, -0.0323029, 0.0169981, -0.0469363, 0.0455998
4: -0.0230227, 0.0060431, -0.0257640, 0.0267363, -0.0497590, 0.0318072

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121846, upper bound: 0.0128077
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121846, upper bound: 0.0123389
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0150241, 0.0040840, -0.0177381, 0.0158506
1: -0.0307293, 0.0087318, -0.0378083, 0.0124536, -0.0431829, 0.0465402
2: -0.0285153, 0.0151063, -0.0293912, 0.0295630, -0.0580783, 0.0444975
3: -0.0298880, 0.0137899, -0.0343575, 0.0203385, -0.0502265, 0.0481473
4: -0.0243416, 0.0171402, -0.0276911, 0.0320904, -0.0564320, 0.0448313

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122289, upper bound: 0.0130628
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122289, upper bound: 0.0130071
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0136542, 0.0008266, -0.0142366, 0.0030378, -0.0166919, 0.0150632
1: -0.0307293, 0.0087318, -0.0347289, 0.0097630, -0.0404923, 0.0434608
2: -0.0285153, 0.0151063, -0.0279771, 0.0244865, -0.0530018, 0.0430834
3: -0.0298880, 0.0137899, -0.0323029, 0.0169981, -0.0468861, 0.0460928
4: -0.0243416, 0.0171402, -0.0257640, 0.0267363, -0.0510779, 0.0429042

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0123122, upper bound: 0.0130708
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0123122, upper bound: 0.0129407
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0151567, 0.0043182, -0.0126664, -0.0000464, -0.0151103, 0.0169846
1: -0.0381850, 0.0127767, -0.0293014, 0.0063529, -0.0445380, 0.0420780
2: -0.0292611, 0.0297494, -0.0269340, 0.0033394, -0.0326005, 0.0566835
3: -0.0349177, 0.0209210, -0.0290113, 0.0116298, -0.0465476, 0.0499323
4: -0.0277269, 0.0324924, -0.0223706, 0.0056239, -0.0333508, 0.0548630

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0143511, 0.0032448, -0.0126664, -0.0000464, -0.0143047, 0.0159113
1: -0.0349527, 0.0099729, -0.0293014, 0.0063529, -0.0413056, 0.0392743
2: -0.0278904, 0.0245293, -0.0269340, 0.0033394, -0.0312298, 0.0514634
3: -0.0325822, 0.0173989, -0.0290113, 0.0116298, -0.0442120, 0.0464101
4: -0.0258082, 0.0269775, -0.0223706, 0.0056239, -0.0314321, 0.0493481

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0151567, 0.0043182, -0.0133985, 0.0006346, -0.0157914, 0.0177167
1: -0.0381850, 0.0127767, -0.0302017, 0.0074749, -0.0456599, 0.0429784
2: -0.0292611, 0.0297494, -0.0277911, 0.0146470, -0.0439081, 0.0575405
3: -0.0349177, 0.0209210, -0.0292889, 0.0126968, -0.0476146, 0.0502100
4: -0.0277269, 0.0324924, -0.0237294, 0.0166738, -0.0444008, 0.0562218

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0115887, upper bound: 0.0125749
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128340, upper bound: 0.0131243
time: 0.21 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.32 seconds
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0108209, upper bound: 0.0132342
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0117404, upper bound: 0.0133315
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0108209, upper bound: 0.0132342
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0117404, upper bound: 0.0133315
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0111201, upper bound: 0.0132678
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0117294, upper bound: 0.0133227
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0111201, upper bound: 0.0132678
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0117294, upper bound: 0.0133227
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122480, upper bound: 0.0133575
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0124492, upper bound: 0.0134548
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122480, upper bound: 0.0133575
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0124492, upper bound: 0.0134548
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0124179, upper bound: 0.0133910
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0125950, upper bound: 0.0134459
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0124179, upper bound: 0.0133910
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0125950, upper bound: 0.0134459
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122578, upper bound: 0.0124649
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122578, upper bound: 0.0124649
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122583, upper bound: 0.0123024
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122583, upper bound: 0.0123024
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123854, upper bound: 0.0128255
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123854, upper bound: 0.0128255
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123860, upper bound: 0.0125001
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123860, upper bound: 0.0125131
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0121846, upper bound: 0.0128077
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0121846, upper bound: 0.0123389
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122289, upper bound: 0.0130628
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0122289, upper bound: 0.0130071
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123122, upper bound: 0.0130708
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0123122, upper bound: 0.0129407
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0115887, upper bound: 0.0125749
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.32
Output dim: 0, lower bound: -0.0128340, upper bound: 0.0131243

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0124079, -0.0000451, -0.0112948, -0.0041046, -0.0083033, 0.0112497
1: -0.0283703, 0.0066859, -0.0196947, -0.0005234, -0.0278468, 0.0263806
2: -0.0267635, 0.0019226, -0.0234008, -0.0066568, -0.0201068, 0.0253234
3: -0.0285679, 0.0117544, -0.0185211, 0.0020294, -0.0305973, 0.0302756
4: -0.0217008, 0.0044460, -0.0194711, -0.0049051, -0.0167956, 0.0239171

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0126128, -0.0001646, -0.0112948, -0.0041046, -0.0085081, 0.0111301
1: -0.0290753, 0.0061500, -0.0196947, -0.0005234, -0.0285519, 0.0258448
2: -0.0268088, 0.0031291, -0.0234008, -0.0066568, -0.0201520, 0.0265299
3: -0.0288192, 0.0113537, -0.0185211, 0.0020294, -0.0308486, 0.0298748
4: -0.0222331, 0.0053785, -0.0194711, -0.0049051, -0.0173280, 0.0248496

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0124079, -0.0000451, -0.0116469, -0.0039406, -0.0084674, 0.0116018
1: -0.0283703, 0.0066859, -0.0199876, -0.0001259, -0.0282444, 0.0266735
2: -0.0267635, 0.0019226, -0.0241192, -0.0032852, -0.0234783, 0.0260417
3: -0.0285679, 0.0117544, -0.0188645, 0.0021902, -0.0307581, 0.0306190
4: -0.0217008, 0.0044460, -0.0200901, -0.0025771, -0.0191237, 0.0245360

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0126128, -0.0001646, -0.0116469, -0.0039406, -0.0086722, 0.0114822
1: -0.0290753, 0.0061500, -0.0199876, -0.0001259, -0.0289494, 0.0261376
2: -0.0268088, 0.0031291, -0.0241192, -0.0032852, -0.0235236, 0.0272482
3: -0.0288192, 0.0113537, -0.0188645, 0.0021902, -0.0310094, 0.0302182
4: -0.0222331, 0.0053785, -0.0200901, -0.0025771, -0.0196560, 0.0254686

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0112948, -0.0041046, -0.0088949, 0.0117008
1: -0.0289199, 0.0061732, -0.0196947, -0.0005234, -0.0283965, 0.0258679
2: -0.0266405, 0.0109326, -0.0234008, -0.0066568, -0.0199837, 0.0343335
3: -0.0282478, 0.0111831, -0.0185211, 0.0020294, -0.0302772, 0.0297042
4: -0.0227388, 0.0129788, -0.0194711, -0.0049051, -0.0178337, 0.0324500

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0112948, -0.0041046, -0.0092457, 0.0118470
1: -0.0299907, 0.0072886, -0.0196947, -0.0005234, -0.0294672, 0.0269833
2: -0.0276908, 0.0144022, -0.0234008, -0.0066568, -0.0210341, 0.0378030
3: -0.0291007, 0.0124004, -0.0185211, 0.0020294, -0.0311301, 0.0309216
4: -0.0236162, 0.0163814, -0.0194711, -0.0049051, -0.0187111, 0.0358525

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0116469, -0.0039406, -0.0090590, 0.0120529
1: -0.0289199, 0.0061732, -0.0199876, -0.0001259, -0.0287940, 0.0261608
2: -0.0266405, 0.0109326, -0.0241192, -0.0032852, -0.0233553, 0.0350518
3: -0.0282478, 0.0111831, -0.0188645, 0.0021902, -0.0304380, 0.0300476
4: -0.0227388, 0.0129788, -0.0200901, -0.0025771, -0.0201617, 0.0330689

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0116469, -0.0039406, -0.0094098, 0.0121991
1: -0.0299907, 0.0072886, -0.0199876, -0.0001259, -0.0298648, 0.0272762
2: -0.0276908, 0.0144022, -0.0241192, -0.0032852, -0.0244056, 0.0385214
3: -0.0291007, 0.0124004, -0.0188645, 0.0021902, -0.0312909, 0.0312650
4: -0.0236162, 0.0163814, -0.0200901, -0.0025771, -0.0210391, 0.0364714

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0124079, -0.0000451, -0.0130033, 0.0001957, -0.0126037, 0.0129583
1: -0.0283703, 0.0066859, -0.0302214, 0.0081966, -0.0365668, 0.0369073
2: -0.0267635, 0.0019226, -0.0280575, 0.0037971, -0.0305607, 0.0299801
3: -0.0285679, 0.0117544, -0.0299382, 0.0132969, -0.0418647, 0.0416926
4: -0.0217008, 0.0044460, -0.0230227, 0.0060431, -0.0277439, 0.0274686

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122306, upper bound: 0.0124490
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122306, upper bound: 0.0124489
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0126128, -0.0001646, -0.0130033, 0.0001957, -0.0128085, 0.0128387
1: -0.0290753, 0.0061500, -0.0302214, 0.0081966, -0.0372719, 0.0363715
2: -0.0268088, 0.0031291, -0.0280575, 0.0037971, -0.0306059, 0.0311866
3: -0.0288192, 0.0113537, -0.0299382, 0.0132969, -0.0421160, 0.0412919
4: -0.0222331, 0.0053785, -0.0230227, 0.0060431, -0.0282763, 0.0284012

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124862, upper bound: 0.0125463
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124862, upper bound: 0.0125463
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0124079, -0.0000451, -0.0136542, 0.0008266, -0.0132345, 0.0136091
1: -0.0283703, 0.0066859, -0.0307293, 0.0087318, -0.0371021, 0.0374152
2: -0.0267635, 0.0019226, -0.0285153, 0.0151063, -0.0418698, 0.0304379
3: -0.0285679, 0.0117544, -0.0298880, 0.0137899, -0.0423577, 0.0416424
4: -0.0217008, 0.0044460, -0.0243416, 0.0171402, -0.0388410, 0.0287875

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122005, upper bound: 0.0127147
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122005, upper bound: 0.0127147
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0126128, -0.0001646, -0.0136542, 0.0008266, -0.0134394, 0.0134895
1: -0.0290753, 0.0061500, -0.0307293, 0.0087318, -0.0378071, 0.0368793
2: -0.0268088, 0.0031291, -0.0285153, 0.0151063, -0.0419151, 0.0316444
3: -0.0288192, 0.0113537, -0.0298880, 0.0137899, -0.0426090, 0.0412416
4: -0.0222331, 0.0053785, -0.0243416, 0.0171402, -0.0393733, 0.0297201

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123832, upper bound: 0.0127707
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123832, upper bound: 0.0127707
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0130033, 0.0001957, -0.0131953, 0.0134094
1: -0.0289199, 0.0061732, -0.0302214, 0.0081966, -0.0371165, 0.0363946
2: -0.0266405, 0.0109326, -0.0280575, 0.0037971, -0.0304376, 0.0389902
3: -0.0282478, 0.0111831, -0.0299382, 0.0132969, -0.0415446, 0.0411213
4: -0.0227388, 0.0129788, -0.0230227, 0.0060431, -0.0287820, 0.0360015

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124070, upper bound: 0.0124825
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124070, upper bound: 0.0124825
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0130033, 0.0001957, -0.0135461, 0.0135556
1: -0.0299907, 0.0072886, -0.0302214, 0.0081966, -0.0381873, 0.0375100
2: -0.0276908, 0.0144022, -0.0280575, 0.0037971, -0.0314880, 0.0424597
3: -0.0291007, 0.0124004, -0.0299382, 0.0132969, -0.0423976, 0.0423386
4: -0.0236162, 0.0163814, -0.0230227, 0.0060431, -0.0296594, 0.0394040

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126081, upper bound: 0.0125374
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126081, upper bound: 0.0134568
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0136542, 0.0008266, -0.0138261, 0.0140602
1: -0.0289199, 0.0061732, -0.0307293, 0.0087318, -0.0376518, 0.0369025
2: -0.0266405, 0.0109326, -0.0285153, 0.0151063, -0.0417468, 0.0394480
3: -0.0282478, 0.0111831, -0.0298880, 0.0137899, -0.0420376, 0.0410711
4: -0.0227388, 0.0129788, -0.0243416, 0.0171402, -0.0398790, 0.0373204

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123947, upper bound: 0.0126106
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123946, upper bound: 0.0126106
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0136542, 0.0008266, -0.0141769, 0.0142064
1: -0.0299907, 0.0072886, -0.0307293, 0.0087318, -0.0387225, 0.0380179
2: -0.0276908, 0.0144022, -0.0285153, 0.0151063, -0.0427971, 0.0429175
3: -0.0291007, 0.0124004, -0.0298880, 0.0137899, -0.0428906, 0.0422884
4: -0.0236162, 0.0163814, -0.0243416, 0.0171402, -0.0407564, 0.0407229

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125450, upper bound: 0.0126106
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125450, upper bound: 0.0126106
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0150241, 0.0040840, -0.0170835, 0.0154301
1: -0.0289199, 0.0061732, -0.0378083, 0.0124536, -0.0413735, 0.0439816
2: -0.0266405, 0.0109326, -0.0293912, 0.0295630, -0.0562035, 0.0403238
3: -0.0282478, 0.0111831, -0.0343575, 0.0203385, -0.0485863, 0.0455405
4: -0.0227388, 0.0129788, -0.0276911, 0.0320904, -0.0548292, 0.0406699

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0150241, 0.0040840, -0.0174343, 0.0155763
1: -0.0299907, 0.0072886, -0.0378083, 0.0124536, -0.0424442, 0.0450970
2: -0.0276908, 0.0144022, -0.0293912, 0.0295630, -0.0572538, 0.0437934
3: -0.0291007, 0.0124004, -0.0343575, 0.0203385, -0.0494392, 0.0467579
4: -0.0236162, 0.0163814, -0.0276911, 0.0320904, -0.0557066, 0.0440725

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129995, 0.0004060, -0.0142366, 0.0030378, -0.0160373, 0.0146426
1: -0.0289199, 0.0061732, -0.0347289, 0.0097630, -0.0386829, 0.0409022
2: -0.0266405, 0.0109326, -0.0279771, 0.0244865, -0.0511270, 0.0389098
3: -0.0282478, 0.0111831, -0.0323029, 0.0169981, -0.0452458, 0.0434860
4: -0.0227388, 0.0129788, -0.0257640, 0.0267363, -0.0494752, 0.0387428

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0142366, 0.0030378, -0.0163881, 0.0147888
1: -0.0299907, 0.0072886, -0.0347289, 0.0097630, -0.0397536, 0.0420175
2: -0.0276908, 0.0144022, -0.0279771, 0.0244865, -0.0521773, 0.0423793
3: -0.0291007, 0.0124004, -0.0323029, 0.0169981, -0.0460988, 0.0447034
4: -0.0236162, 0.0163814, -0.0257640, 0.0267363, -0.0503525, 0.0421454

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151490, 0.0043106, -0.0133985, 0.0006346, -0.0157837, 0.0177091
1: -0.0381647, 0.0127266, -0.0302017, 0.0074749, -0.0456395, 0.0429283
2: -0.0292092, 0.0297307, -0.0277911, 0.0146470, -0.0438561, 0.0575218
3: -0.0348929, 0.0208739, -0.0292889, 0.0126968, -0.0475897, 0.0501628
4: -0.0277078, 0.0324711, -0.0237294, 0.0166738, -0.0443816, 0.0562005

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125808, upper bound: 0.0129171
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125808, upper bound: 0.0129171
time: 0.23 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.36 seconds
NS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0122306, upper bound: 0.0124490
NS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0122306, upper bound: 0.0124489
NS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0124862, upper bound: 0.0125463
NS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0124862, upper bound: 0.0125463
NS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0122005, upper bound: 0.0127147
NS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0122005, upper bound: 0.0127147
NS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0123832, upper bound: 0.0127707
NS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0123832, upper bound: 0.0127707
NS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0124070, upper bound: 0.0124825
NS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0124070, upper bound: 0.0124825
NS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0126081, upper bound: 0.0125374
NS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0126081, upper bound: 0.0134568
NS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0123947, upper bound: 0.0126106
NS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0123946, upper bound: 0.0126106
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0125450, upper bound: 0.0126106
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0125450, upper bound: 0.0126106
NS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0125808, upper bound: 0.0129171
NS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0125808, upper bound: 0.0129171

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133504, 0.0005522, -0.0126128, -0.0001646, -0.0131857, 0.0131650
1: -0.0299907, 0.0072886, -0.0290753, 0.0061500, -0.0361407, 0.0363639
2: -0.0276908, 0.0144022, -0.0268088, 0.0031291, -0.0308199, 0.0412110
3: -0.0291007, 0.0124004, -0.0288192, 0.0113537, -0.0404544, 0.0412196
4: -0.0236162, 0.0163814, -0.0222331, 0.0053785, -0.0289947, 0.0386145

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123778, upper bound: 0.0120603
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126050, upper bound: 0.0125345
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151490, 0.0043106, -0.0133546, 0.0006833, -0.0158324, 0.0176652
1: -0.0381647, 0.0127266, -0.0302651, 0.0076143, -0.0457789, 0.0429918
2: -0.0292092, 0.0297307, -0.0276898, 0.0150388, -0.0442479, 0.0574205
3: -0.0348929, 0.0208739, -0.0291658, 0.0128613, -0.0477542, 0.0500397
4: -0.0277078, 0.0324711, -0.0237524, 0.0170569, -0.0447647, 0.0562235

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151490, 0.0043106, -0.0132060, 0.0004814, -0.0156304, 0.0175166
1: -0.0381647, 0.0127266, -0.0296921, 0.0070918, -0.0452565, 0.0424187
2: -0.0292092, 0.0297307, -0.0273984, 0.0138360, -0.0430452, 0.0571291
3: -0.0348929, 0.0208739, -0.0288538, 0.0122143, -0.0471072, 0.0497277
4: -0.0277078, 0.0324711, -0.0233524, 0.0157887, -0.0434965, 0.0558235

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.54 + 101.59 = 103.13 seconds
