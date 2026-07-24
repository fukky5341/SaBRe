## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.3265e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010831, 0.0010990, 0.0010831, 0.0010990, -0.0000135, 0.0000135)
1: (0.9936353, 0.9937307, 0.9936353, 0.9937307, -0.0000657, 0.0000657)
2: (-0.0063962, -0.0056234, -0.0063962, -0.0056234, -0.0005023, 0.0005023)
3: (0.0039441, 0.0040031, 0.0039441, 0.0040031, -0.0000389, 0.0000389)
4: (0.0028615, 0.0034723, 0.0028615, 0.0034723, -0.0004005, 0.0004005)
5: (0.0062138, 0.0063808, 0.0062138, 0.0063808, -0.0001280, 0.0001280)
6: (-0.0013413, -0.0010730, -0.0013413, -0.0010730, -0.0001741, 0.0001741)
7: (-0.0082082, -0.0080621, -0.0082082, -0.0080621, -0.0001092, 0.0001092)
8: (0.0056401, 0.0066555, 0.0056401, 0.0066555, -0.0006496, 0.0006496)
9: (-0.0036815, -0.0036476, -0.0036815, -0.0036476, -0.0000339, 0.0000339)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.28 = 2.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0000571, upper bound: 0.0000570

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000563, upper bound: 0.0000563
time: 0.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000563, upper bound: 0.0000563
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 1, lower bound: -0.0000563, upper bound: 0.0000563
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 1, lower bound: -0.0000563, upper bound: 0.0000563

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0010834, 0.0010990, 0.0010831, 0.0010990, -0.0000132, 0.0000134
1: 0.9936361, 0.9937298, 0.9936354, 0.9937307, -0.0000649, 0.0000650
2: -0.0063877, -0.0056247, -0.0063958, -0.0056235, -0.0004943, 0.0005012
3: 0.0039444, 0.0040026, 0.0039441, 0.0040031, -0.0000386, 0.0000383
4: 0.0028625, 0.0034655, 0.0028615, 0.0034719, -0.0003997, 0.0003943
5: 0.0062156, 0.0063806, 0.0062139, 0.0063808, -0.0001260, 0.0001278
6: -0.0013383, -0.0010735, -0.0013411, -0.0010731, -0.0001713, 0.0001738
7: -0.0082080, -0.0080637, -0.0082082, -0.0080622, -0.0001090, 0.0001075
8: 0.0056418, 0.0066443, 0.0056402, 0.0066550, -0.0006481, 0.0006381
9: -0.0036815, -0.0036485, -0.0036815, -0.0036476, -0.0000338, 0.0000330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0010831, 0.0010990, 0.0010831, 0.0010990, -0.0000135, 0.0000134
1: 0.9936361, 0.9937305, 0.9936354, 0.9937307, -0.0000649, 0.0000659
2: -0.0063877, -0.0056235, -0.0063957, -0.0056235, -0.0004941, 0.0005038
3: 0.0039441, 0.0040026, 0.0039441, 0.0040031, -0.0000391, 0.0000383
4: 0.0028616, 0.0034655, 0.0028615, 0.0034719, -0.0004017, 0.0003942
5: 0.0062157, 0.0063808, 0.0062139, 0.0063808, -0.0001266, 0.0001283
6: -0.0013383, -0.0010731, -0.0013411, -0.0010731, -0.0001713, 0.0001746
7: -0.0082082, -0.0080637, -0.0082082, -0.0080622, -0.0001094, 0.0001078
8: 0.0056403, 0.0066442, 0.0056402, 0.0066549, -0.0006517, 0.0006384
9: -0.0036815, -0.0036477, -0.0036815, -0.0036477, -0.0000338, 0.0000338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.24 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010831, 0.0010990, -0.0000077, 0.0000134
1: 0.9936361, 0.9937159, 0.9936354, 0.9937307, -0.0000649, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056235, -0.0004884, 0.0004661
3: 0.0039502, 0.0040026, 0.0039441, 0.0040031, -0.0000329, 0.0000383
4: 0.0028817, 0.0034655, 0.0028615, 0.0034719, -0.0003719, 0.0003874
5: 0.0062156, 0.0063753, 0.0062139, 0.0063808, -0.0001126, 0.0001202
6: -0.0013383, -0.0010819, -0.0013411, -0.0010731, -0.0001694, 0.0001616
7: -0.0082034, -0.0080637, -0.0082082, -0.0080622, -0.0001023, 0.0000974
8: 0.0056737, 0.0066443, 0.0056402, 0.0066550, -0.0006020, 0.0006373
9: -0.0036812, -0.0036662, -0.0036815, -0.0036476, -0.0000336, 0.0000153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81
type: B, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010869, 0.0010990, -0.0000034, 0.0000105
1: 0.9936334, 0.9936987, 0.9936354, 0.9937209, -0.0000671, 0.0000426
2: -0.0064159, -0.0057297, -0.0063958, -0.0056402, -0.0005928, 0.0004531
3: 0.0039590, 0.0040044, 0.0039481, 0.0040031, -0.0000297, 0.0000429
4: 0.0029455, 0.0034878, 0.0028747, 0.0034719, -0.0003601, 0.0004685
5: 0.0062096, 0.0063579, 0.0062139, 0.0063772, -0.0001281, 0.0001091
6: -0.0013481, -0.0011099, -0.0013412, -0.0010789, -0.0002058, 0.0001572
7: -0.0081881, -0.0080584, -0.0082050, -0.0080621, -0.0000938, 0.0001121
8: 0.0057798, 0.0066813, 0.0056622, 0.0066550, -0.0005893, 0.0007788
9: -0.0036804, -0.0036734, -0.0036813, -0.0036599, -0.0000206, 0.0000061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010831, 0.0010990, -0.0000078, 0.0000134
1: 0.9936361, 0.9937163, 0.9936354, 0.9937307, -0.0000649, 0.0000518
2: -0.0063877, -0.0056485, -0.0063957, -0.0056235, -0.0004883, 0.0004681
3: 0.0039501, 0.0040026, 0.0039441, 0.0040031, -0.0000332, 0.0000383
4: 0.0028813, 0.0034655, 0.0028615, 0.0034719, -0.0003735, 0.0003873
5: 0.0062157, 0.0063754, 0.0062139, 0.0063808, -0.0001126, 0.0001206
6: -0.0013383, -0.0010817, -0.0013411, -0.0010731, -0.0001694, 0.0001623
7: -0.0082035, -0.0080637, -0.0082082, -0.0080622, -0.0001027, 0.0000975
8: 0.0056730, 0.0066442, 0.0056402, 0.0066549, -0.0006048, 0.0006376
9: -0.0036812, -0.0036659, -0.0036815, -0.0036477, -0.0000336, 0.0000156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81
type: B, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010866, 0.0010990, -0.0000034, 0.0000109
1: 0.9936332, 0.9936988, 0.9936354, 0.9937219, -0.0000679, 0.0000429
2: -0.0064186, -0.0057278, -0.0063957, -0.0056386, -0.0005936, 0.0004573
3: 0.0039589, 0.0040046, 0.0039477, 0.0040031, -0.0000299, 0.0000431
4: 0.0029440, 0.0034899, 0.0028735, 0.0034719, -0.0003635, 0.0004691
5: 0.0062090, 0.0063583, 0.0062139, 0.0063775, -0.0001283, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010783, -0.0002060, 0.0001586
7: -0.0081885, -0.0080578, -0.0082053, -0.0080622, -0.0000951, 0.0001122
8: 0.0057773, 0.0066849, 0.0056601, 0.0066549, -0.0005943, 0.0007799
9: -0.0036804, -0.0036734, -0.0036813, -0.0036587, -0.0000217, 0.0000061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.33 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010887, 0.0010990, -0.0000077, 0.0000078
1: 0.9936361, 0.9937159, 0.9936354, 0.9937164, -0.0000509, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004605
3: 0.0039502, 0.0040026, 0.0039500, 0.0040031, -0.0000329, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036658, -0.0000155, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.46 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.46 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010887, 0.0010990, -0.0000078, 0.0000078
1: 0.9936361, 0.9937163, 0.9936354, 0.9937164, -0.0000509, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004627
3: 0.0039501, 0.0040026, 0.0039500, 0.0040031, -0.0000332, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036658, -0.0000155, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.46 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.20 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010887, 0.0010990, -0.0000077, 0.0000078
1: 0.9936361, 0.9937159, 0.9936354, 0.9937164, -0.0000509, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004605
3: 0.0039502, 0.0040026, 0.0039500, 0.0040031, -0.0000329, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036658, -0.0000155, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010887, 0.0010990, -0.0000078, 0.0000078
1: 0.9936361, 0.9937163, 0.9936354, 0.9937164, -0.0000509, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004627
3: 0.0039501, 0.0040026, 0.0039500, 0.0040031, -0.0000332, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036658, -0.0000155, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.33 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010887, 0.0010990, -0.0000077, 0.0000078
1: 0.9936361, 0.9937159, 0.9936354, 0.9937164, -0.0000509, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004605
3: 0.0039502, 0.0040026, 0.0039500, 0.0040031, -0.0000329, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036658, -0.0000155, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010934, 0.0010990, -0.0000077, 0.0000034
1: 0.9936361, 0.9937159, 0.9936354, 0.9937063, -0.0000427, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004572
3: 0.0039502, 0.0040026, 0.0039536, 0.0040031, -0.0000329, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036736, -0.0000047, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.47 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010887, 0.0010990, -0.0000078, 0.0000078
1: 0.9936361, 0.9937163, 0.9936354, 0.9937164, -0.0000509, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004627
3: 0.0039501, 0.0040026, 0.0039500, 0.0040031, -0.0000332, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036658, -0.0000155, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010934, 0.0010990, -0.0000078, 0.0000034
1: 0.9936361, 0.9937163, 0.9936354, 0.9937063, -0.0000427, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004594
3: 0.0039501, 0.0040026, 0.0039536, 0.0040031, -0.0000332, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036736, -0.0000047, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.37 seconds
IS_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010887, 0.0010990, -0.0000077, 0.0000078
1: 0.9936361, 0.9937159, 0.9936354, 0.9937164, -0.0000509, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004605
3: 0.0039502, 0.0040026, 0.0039500, 0.0040031, -0.0000329, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036658, -0.0000155, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010934, 0.0010990, -0.0000077, 0.0000034
1: 0.9936361, 0.9937159, 0.9936354, 0.9937063, -0.0000427, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004572
3: 0.0039502, 0.0040026, 0.0039536, 0.0040031, -0.0000329, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036736, -0.0000047, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010887, 0.0010990, -0.0000078, 0.0000078
1: 0.9936361, 0.9937163, 0.9936354, 0.9937164, -0.0000509, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004627
3: 0.0039501, 0.0040026, 0.0039500, 0.0040031, -0.0000332, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036658, -0.0000155, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010934, 0.0010990, -0.0000078, 0.0000034
1: 0.9936361, 0.9937163, 0.9936354, 0.9937063, -0.0000427, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004594
3: 0.0039501, 0.0040026, 0.0039536, 0.0040031, -0.0000332, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036736, -0.0000047, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.40 seconds
IS_A1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000512, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.40
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## BFS IS instance: IS_A1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010887, 0.0010990, -0.0000077, 0.0000078
1: 0.9936361, 0.9937159, 0.9936354, 0.9937164, -0.0000509, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004605
3: 0.0039502, 0.0040026, 0.0039500, 0.0040031, -0.0000329, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036658, -0.0000155, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.55 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010934, 0.0010990, -0.0000077, 0.0000034
1: 0.9936361, 0.9937159, 0.9936354, 0.9937063, -0.0000427, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004572
3: 0.0039502, 0.0040026, 0.0039536, 0.0040031, -0.0000329, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036736, -0.0000047, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010934, 0.0010990, -0.0000077, 0.0000034
1: 0.9936361, 0.9937159, 0.9936354, 0.9937063, -0.0000427, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004572
3: 0.0039502, 0.0040026, 0.0039536, 0.0040031, -0.0000329, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036736, -0.0000047, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010934, 0.0010990, -0.0000077, 0.0000034
1: 0.9936361, 0.9937159, 0.9936354, 0.9937063, -0.0000427, 0.0000513
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004527, 0.0004572
3: 0.0039502, 0.0040026, 0.0039536, 0.0040031, -0.0000329, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003592
5: 0.0062156, 0.0063753, 0.0062139, 0.0063755, -0.0001049, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001571, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000907
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005904
9: -0.0036812, -0.0036662, -0.0036812, -0.0036736, -0.0000047, 0.0000150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010889, 0.0010990, 0.0010940, 0.0010993, -0.0000086, 0.0000035
1: 0.9936361, 0.9937159, 0.9936323, 0.9936987, -0.0000443, 0.0000624
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004705, 0.0005737
3: 0.0039502, 0.0040026, 0.0039590, 0.0040052, -0.0000406, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003733
5: 0.0062156, 0.0063753, 0.0062071, 0.0063581, -0.0001087, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001632, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000940
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006138
9: -0.0036812, -0.0036662, -0.0036804, -0.0036733, -0.0000059, 0.0000142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004605
3: 0.0039537, 0.0040026, 0.0039500, 0.0040031, -0.0000303, 0.0000325
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003652, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0001065
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001598
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000922, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006012, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.55 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936334, 0.9936987, 0.9936354, 0.9937164, -0.0000617, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004775
3: 0.0039590, 0.0040044, 0.0039500, 0.0040031, -0.0000314, 0.0000400
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003787, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001102
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001657
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000954, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006236, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.54 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.55 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.58 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000434
2: -0.0063877, -0.0056490, -0.0063958, -0.0056483, -0.0004489, 0.0004572
3: 0.0039537, 0.0040026, 0.0039536, 0.0040031, -0.0000303, 0.0000297
4: 0.0028817, 0.0034655, 0.0028811, 0.0034719, -0.0003614, 0.0003548
5: 0.0062157, 0.0063753, 0.0062139, 0.0063755, -0.0000970, 0.0000988
6: -0.0013383, -0.0010819, -0.0013411, -0.0010817, -0.0001558, 0.0001587
7: -0.0082034, -0.0080637, -0.0082035, -0.0080622, -0.0000864, 0.0000849
8: 0.0056737, 0.0066443, 0.0056728, 0.0066550, -0.0006007, 0.0005899
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000043, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000443, 0.0000545
2: -0.0063877, -0.0056490, -0.0064273, -0.0057288, -0.0004667, 0.0005737
3: 0.0039537, 0.0040026, 0.0039590, 0.0040052, -0.0000380, 0.0000309
4: 0.0028817, 0.0034655, 0.0029448, 0.0034968, -0.0004534, 0.0003689
5: 0.0062157, 0.0063753, 0.0062071, 0.0063581, -0.0001009, 0.0001240
6: -0.0013383, -0.0010819, -0.0013521, -0.0011096, -0.0001620, 0.0001991
7: -0.0082034, -0.0080637, -0.0081883, -0.0080562, -0.0001085, 0.0000883
8: 0.0056737, 0.0066443, 0.0057786, 0.0066963, -0.0007538, 0.0006132
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936334, 0.9936987, 0.9936354, 0.9937063, -0.0000535, 0.0000451
2: -0.0064159, -0.0057297, -0.0063958, -0.0056483, -0.0005633, 0.0004742
3: 0.0039590, 0.0040044, 0.0039536, 0.0040031, -0.0000314, 0.0000373
4: 0.0029455, 0.0034878, 0.0028811, 0.0034719, -0.0003748, 0.0004452
5: 0.0062096, 0.0063579, 0.0062139, 0.0063755, -0.0001217, 0.0001025
6: -0.0013481, -0.0011099, -0.0013411, -0.0010817, -0.0001955, 0.0001646
7: -0.0081881, -0.0080584, -0.0082035, -0.0080622, -0.0000897, 0.0001065
8: 0.0057798, 0.0066813, 0.0056728, 0.0066550, -0.0006231, 0.0007401
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936334, 0.9936987, 0.9936323, 0.9936987, -0.0000415, 0.0000426
2: -0.0064159, -0.0057297, -0.0064273, -0.0057288, -0.0004370, 0.0004479
3: 0.0039590, 0.0040044, 0.0039590, 0.0040052, -0.0000297, 0.0000289
4: 0.0029455, 0.0034878, 0.0029448, 0.0034968, -0.0003540, 0.0003454
5: 0.0062096, 0.0063579, 0.0062071, 0.0063581, -0.0000945, 0.0000968
6: -0.0013481, -0.0011099, -0.0013521, -0.0011096, -0.0001517, 0.0001555
7: -0.0081881, -0.0080584, -0.0081883, -0.0080562, -0.0000847, 0.0000826
8: 0.0057798, 0.0066813, 0.0057786, 0.0066963, -0.0005885, 0.0005742
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010887, 0.0010990, -0.0000078, 0.0000078
1: 0.9936361, 0.9937163, 0.9936354, 0.9937164, -0.0000509, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004627
3: 0.0039501, 0.0040026, 0.0039500, 0.0040031, -0.0000332, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036658, -0.0000155, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010934, 0.0010990, -0.0000078, 0.0000034
1: 0.9936361, 0.9937163, 0.9936354, 0.9937063, -0.0000427, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004594
3: 0.0039501, 0.0040026, 0.0039536, 0.0040031, -0.0000332, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036736, -0.0000047, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010934, 0.0010990, -0.0000078, 0.0000034
1: 0.9936361, 0.9937163, 0.9936354, 0.9937063, -0.0000427, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004594
3: 0.0039501, 0.0040026, 0.0039536, 0.0040031, -0.0000332, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036736, -0.0000047, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010934, 0.0010990, -0.0000078, 0.0000034
1: 0.9936361, 0.9937163, 0.9936354, 0.9937063, -0.0000427, 0.0000518
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004527, 0.0004594
3: 0.0039501, 0.0040026, 0.0039536, 0.0040031, -0.0000332, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003591
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0001049, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001570, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000907
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005908
9: -0.0036812, -0.0036659, -0.0036812, -0.0036736, -0.0000047, 0.0000154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010888, 0.0010990, 0.0010940, 0.0010993, -0.0000087, 0.0000035
1: 0.9936361, 0.9937163, 0.9936323, 0.9936987, -0.0000444, 0.0000629
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004705, 0.0005759
3: 0.0039501, 0.0040026, 0.0039590, 0.0040052, -0.0000409, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003732
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001088, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001632, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000941
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006142
9: -0.0036812, -0.0036659, -0.0036804, -0.0036733, -0.0000059, 0.0000145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.55 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010887, 0.0010990, -0.0000035, 0.0000078
1: 0.9936361, 0.9937063, 0.9936354, 0.9937164, -0.0000509, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004627
3: 0.0039536, 0.0040026, 0.0039500, 0.0040031, -0.0000304, 0.0000325
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003669, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0001069
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001605
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000925, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006041, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036658, -0.0000155, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010887, 0.0010990, -0.0000036, 0.0000086
1: 0.9936332, 0.9936988, 0.9936354, 0.9937164, -0.0000618, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004798
3: 0.0039589, 0.0040046, 0.0039500, 0.0040031, -0.0000315, 0.0000401
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003805, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001107
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001665
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000958, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006266, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036658, -0.0000147, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.57 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010940, 0.0010993, -0.0000034, 0.0000033
1: 0.9936332, 0.9936988, 0.9936323, 0.9936987, -0.0000416, 0.0000429
2: -0.0064186, -0.0057278, -0.0064274, -0.0057288, -0.0004378, 0.0004517
3: 0.0039589, 0.0040046, 0.0039590, 0.0040052, -0.0000299, 0.0000290
4: 0.0029440, 0.0034899, 0.0029448, 0.0034969, -0.0003570, 0.0003460
5: 0.0062090, 0.0063583, 0.0062071, 0.0063581, -0.0000946, 0.0000976
6: -0.0013490, -0.0011093, -0.0013521, -0.0011096, -0.0001520, 0.0001568
7: -0.0081885, -0.0080578, -0.0081883, -0.0080562, -0.0000854, 0.0000828
8: 0.0057773, 0.0066849, 0.0057786, 0.0066964, -0.0005935, 0.0005753
9: -0.0036804, -0.0036734, -0.0036804, -0.0036733, -0.0000046, 0.0000045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010934, 0.0010990, -0.0000035, 0.0000034
1: 0.9936361, 0.9937063, 0.9936354, 0.9937063, -0.0000427, 0.0000436
2: -0.0063877, -0.0056485, -0.0063958, -0.0056483, -0.0004493, 0.0004594
3: 0.0039536, 0.0040026, 0.0039536, 0.0040031, -0.0000304, 0.0000297
4: 0.0028813, 0.0034655, 0.0028811, 0.0034719, -0.0003631, 0.0003551
5: 0.0062157, 0.0063754, 0.0062139, 0.0063755, -0.0000971, 0.0000993
6: -0.0013383, -0.0010817, -0.0013411, -0.0010817, -0.0001559, 0.0001595
7: -0.0082035, -0.0080637, -0.0082035, -0.0080622, -0.0000869, 0.0000849
8: 0.0056730, 0.0066442, 0.0056728, 0.0066549, -0.0006036, 0.0005903
9: -0.0036812, -0.0036737, -0.0036812, -0.0036736, -0.0000047, 0.0000046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010934, 0.0010990, 0.0010940, 0.0010993, -0.0000044, 0.0000035
1: 0.9936361, 0.9937063, 0.9936323, 0.9936987, -0.0000444, 0.0000547
2: -0.0063877, -0.0056485, -0.0064274, -0.0057288, -0.0004671, 0.0005759
3: 0.0039536, 0.0040026, 0.0039590, 0.0040052, -0.0000381, 0.0000309
4: 0.0028813, 0.0034655, 0.0029448, 0.0034969, -0.0004551, 0.0003691
5: 0.0062157, 0.0063754, 0.0062071, 0.0063581, -0.0001009, 0.0001245
6: -0.0013383, -0.0010817, -0.0013521, -0.0011096, -0.0001621, 0.0001999
7: -0.0082035, -0.0080637, -0.0081883, -0.0080562, -0.0001089, 0.0000883
8: 0.0056730, 0.0066442, 0.0057786, 0.0066964, -0.0007566, 0.0006137
9: -0.0036812, -0.0036737, -0.0036804, -0.0036733, -0.0000059, 0.0000048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.54 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010940, 0.0010992, 0.0010934, 0.0010990, -0.0000036, 0.0000043
1: 0.9936332, 0.9936988, 0.9936354, 0.9937063, -0.0000535, 0.0000453
2: -0.0064186, -0.0057278, -0.0063958, -0.0056483, -0.0005636, 0.0004766
3: 0.0039589, 0.0040046, 0.0039536, 0.0040031, -0.0000315, 0.0000373
4: 0.0029440, 0.0034899, 0.0028811, 0.0034719, -0.0003766, 0.0004454
5: 0.0062090, 0.0063583, 0.0062139, 0.0063755, -0.0001218, 0.0001030
6: -0.0013490, -0.0011093, -0.0013411, -0.0010817, -0.0001956, 0.0001654
7: -0.0081885, -0.0080578, -0.0082035, -0.0080622, -0.0000901, 0.0001066
8: 0.0057773, 0.0066849, 0.0056728, 0.0066549, -0.0006261, 0.0007405
9: -0.0036804, -0.0036734, -0.0036812, -0.0036736, -0.0000049, 0.0000058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: A, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: A, layer: 3, pos: 136
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
time: 0.55 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.73 seconds
IS_A1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000512
IS_A1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000469, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000512
IS_A2_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472
IS_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 1, lower bound: -0.0000472, upper bound: 0.0000472

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.46 + 597.98 = 600.44 seconds
