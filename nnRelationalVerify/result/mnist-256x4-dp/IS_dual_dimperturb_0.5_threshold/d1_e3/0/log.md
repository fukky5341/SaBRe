## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0005928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0026885, 1.0047075, 1.0026885, 1.0047075, -0.0011629, 0.0011629)
1: (-0.0005940, -0.0000910, -0.0005940, -0.0000910, -0.0002898, 0.0002898)
2: (-0.0095717, -0.0069059, -0.0095717, -0.0069059, -0.0015356, 0.0015356)
3: (0.0018701, 0.0030835, 0.0018701, 0.0030835, -0.0006989, 0.0006989)
4: (-0.0013247, -0.0008087, -0.0013247, -0.0008087, -0.0002972, 0.0002972)
5: (-0.0130792, -0.0097264, -0.0130792, -0.0097264, -0.0019314, 0.0019314)
6: (0.0040095, 0.0048605, 0.0040095, 0.0048605, -0.0004902, 0.0004902)
7: (0.0072361, 0.0094379, 0.0072361, 0.0094379, -0.0012683, 0.0012683)
8: (0.0042413, 0.0053992, 0.0042413, 0.0053992, -0.0006670, 0.0006670)
9: (-0.0081245, -0.0067818, -0.0081245, -0.0067818, -0.0007734, 0.0007734)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.57 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0007299, upper bound: 0.0007299

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006894, upper bound: 0.0006001
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0006894, upper bound: 0.0006001
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0006894, upper bound: 0.0006894

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 1.0026911, 1.0045173, 1.0026892, 1.0046548, -0.0011049, 0.0009658
1: -0.0005934, -0.0001384, -0.0005939, -0.0001041, -0.0002753, 0.0002406
2: -0.0093207, -0.0069093, -0.0095022, -0.0069068, -0.0012753, 0.0014591
3: 0.0018717, 0.0029693, 0.0018706, 0.0030519, -0.0006641, 0.0005805
4: -0.0012761, -0.0008094, -0.0013112, -0.0008089, -0.0002468, 0.0002824
5: -0.0127636, -0.0097306, -0.0129918, -0.0097275, -0.0016040, 0.0018351
6: 0.0040106, 0.0047804, 0.0040098, 0.0048383, -0.0004658, 0.0004071
7: 0.0072389, 0.0092306, 0.0072369, 0.0093805, -0.0012051, 0.0010533
8: 0.0042427, 0.0052901, 0.0042417, 0.0053690, -0.0006337, 0.0005539
9: -0.0079980, -0.0067835, -0.0080894, -0.0067823, -0.0006423, 0.0007349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006001
time: 0.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006001
time: 0.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 1.0025338, 1.0046316, 1.0026895, 1.0046828, -0.0013449, 0.0010390
1: -0.0006326, -0.0001098, -0.0005938, -0.0000971, -0.0003351, 0.0002589
2: -0.0094719, -0.0067015, -0.0095393, -0.0069071, -0.0013720, 0.0017759
3: 0.0017771, 0.0030381, 0.0018707, 0.0030687, -0.0008083, 0.0006245
4: -0.0013054, -0.0007692, -0.0013184, -0.0008090, -0.0002655, 0.0003437
5: -0.0129537, -0.0094693, -0.0130384, -0.0097278, -0.0017256, 0.0022336
6: 0.0039442, 0.0048286, 0.0040099, 0.0048501, -0.0005669, 0.0004380
7: 0.0070673, 0.0093555, 0.0072371, 0.0094111, -0.0014668, 0.0011332
8: 0.0041525, 0.0053558, 0.0042418, 0.0053851, -0.0007714, 0.0005959
9: -0.0080742, -0.0066789, -0.0081081, -0.0067824, -0.0006910, 0.0008944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0006894
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0006894
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006001
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006001
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0006894
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0006894

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026911, 1.0045173, 1.0026911, 1.0045173, -0.0009641, 0.0009641
1: -0.0005934, -0.0001384, -0.0005934, -0.0001384, -0.0002402, 0.0002402
2: -0.0093207, -0.0069093, -0.0093207, -0.0069093, -0.0012731, 0.0012731
3: 0.0018717, 0.0029693, 0.0018717, 0.0029693, -0.0005795, 0.0005795
4: -0.0012761, -0.0008094, -0.0012761, -0.0008094, -0.0002464, 0.0002464
5: -0.0127636, -0.0097306, -0.0127636, -0.0097306, -0.0016013, 0.0016013
6: 0.0040106, 0.0047804, 0.0040106, 0.0047804, -0.0004064, 0.0004064
7: 0.0072389, 0.0092306, 0.0072389, 0.0092306, -0.0010515, 0.0010515
8: 0.0042427, 0.0052901, 0.0042427, 0.0052901, -0.0005530, 0.0005530
9: -0.0079980, -0.0067835, -0.0079980, -0.0067835, -0.0006412, 0.0006412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005614, upper bound: 0.0005624
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005890, upper bound: 0.0005744
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026911, 1.0045173, 1.0025338, 1.0046316, -0.0011148, 0.0011664
1: -0.0005934, -0.0001384, -0.0006326, -0.0001098, -0.0002778, 0.0002906
2: -0.0093207, -0.0069093, -0.0094719, -0.0067015, -0.0015402, 0.0014720
3: 0.0018717, 0.0029693, 0.0017771, 0.0030381, -0.0006700, 0.0007010
4: -0.0012761, -0.0008094, -0.0013054, -0.0007692, -0.0002981, 0.0002849
5: -0.0127636, -0.0097306, -0.0129537, -0.0094693, -0.0019372, 0.0018514
6: 0.0040106, 0.0047804, 0.0039442, 0.0048286, -0.0004699, 0.0004917
7: 0.0072389, 0.0092306, 0.0070673, 0.0093555, -0.0012158, 0.0012721
8: 0.0042427, 0.0052901, 0.0041525, 0.0053558, -0.0006394, 0.0006690
9: -0.0079980, -0.0067835, -0.0080742, -0.0066789, -0.0007757, 0.0007414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005780, upper bound: 0.0005496
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005890, upper bound: 0.0005744
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025338, 1.0046316, 1.0026911, 1.0045173, -0.0011664, 0.0011148
1: -0.0006326, -0.0001098, -0.0005934, -0.0001384, -0.0002906, 0.0002778
2: -0.0094719, -0.0067015, -0.0093207, -0.0069093, -0.0014720, 0.0015402
3: 0.0017771, 0.0030381, 0.0018717, 0.0029693, -0.0007010, 0.0006700
4: -0.0013054, -0.0007692, -0.0012761, -0.0008094, -0.0002849, 0.0002981
5: -0.0129537, -0.0094693, -0.0127636, -0.0097306, -0.0018514, 0.0019372
6: 0.0039442, 0.0048286, 0.0040106, 0.0047804, -0.0004917, 0.0004699
7: 0.0070673, 0.0093555, 0.0072389, 0.0092306, -0.0012721, 0.0012158
8: 0.0041525, 0.0053558, 0.0042427, 0.0052901, -0.0006690, 0.0006394
9: -0.0080742, -0.0066789, -0.0079980, -0.0067835, -0.0007414, 0.0007757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005496, upper bound: 0.0006525
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006641
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025338, 1.0046316, 1.0025338, 1.0046316, -0.0010516, 0.0010516
1: -0.0006326, -0.0001098, -0.0006326, -0.0001098, -0.0002620, 0.0002620
2: -0.0094719, -0.0067015, -0.0094719, -0.0067015, -0.0013886, 0.0013886
3: 0.0017771, 0.0030381, 0.0017771, 0.0030381, -0.0006321, 0.0006321
4: -0.0013054, -0.0007692, -0.0013054, -0.0007692, -0.0002688, 0.0002688
5: -0.0129537, -0.0094693, -0.0129537, -0.0094693, -0.0017466, 0.0017466
6: 0.0039442, 0.0048286, 0.0039442, 0.0048286, -0.0004433, 0.0004433
7: 0.0070673, 0.0093555, 0.0070673, 0.0093555, -0.0011469, 0.0011469
8: 0.0041525, 0.0053558, 0.0041525, 0.0053558, -0.0006032, 0.0006032
9: -0.0080742, -0.0066789, -0.0080742, -0.0066789, -0.0006994, 0.0006994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005496, upper bound: 0.0006525
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006641
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005614, upper bound: 0.0005624
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005890, upper bound: 0.0005744
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005780, upper bound: 0.0005496
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005890, upper bound: 0.0005744
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005496, upper bound: 0.0006525
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006641
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005496, upper bound: 0.0006525
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006641

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0047201, 1.0027256, 1.0045161, -0.0010034, 0.0011167
1: -0.0005964, -0.0000879, -0.0005848, -0.0001387, -0.0002500, 0.0002783
2: -0.0095884, -0.0068935, -0.0093191, -0.0069549, -0.0014746, 0.0013250
3: 0.0018645, 0.0030911, 0.0018925, 0.0029685, -0.0006031, 0.0006712
4: -0.0013279, -0.0008063, -0.0012758, -0.0008182, -0.0002854, 0.0002565
5: -0.0131003, -0.0097108, -0.0127615, -0.0097880, -0.0018547, 0.0016665
6: 0.0040055, 0.0048658, 0.0040251, 0.0047798, -0.0004230, 0.0004707
7: 0.0072259, 0.0094518, 0.0072766, 0.0092293, -0.0010944, 0.0012179
8: 0.0042359, 0.0054064, 0.0042626, 0.0052894, -0.0005755, 0.0006405
9: -0.0081329, -0.0067756, -0.0079972, -0.0068065, -0.0007427, 0.0006673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0046310, 1.0026911, 1.0045173, -0.0010323, 0.0011101
1: -0.0006289, -0.0001100, -0.0005934, -0.0001384, -0.0002572, 0.0002766
2: -0.0094709, -0.0067212, -0.0093207, -0.0069093, -0.0014659, 0.0013632
3: 0.0017861, 0.0030376, 0.0018717, 0.0029693, -0.0006204, 0.0006672
4: -0.0013052, -0.0007730, -0.0012761, -0.0008094, -0.0002837, 0.0002638
5: -0.0129525, -0.0094941, -0.0127636, -0.0097306, -0.0018437, 0.0017145
6: 0.0039505, 0.0048283, 0.0040106, 0.0047804, -0.0004352, 0.0004680
7: 0.0070836, 0.0093547, 0.0072389, 0.0092306, -0.0011259, 0.0012107
8: 0.0041610, 0.0053554, 0.0042427, 0.0052901, -0.0005921, 0.0006367
9: -0.0080737, -0.0066888, -0.0079980, -0.0067835, -0.0007383, 0.0006866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005206, upper bound: 0.0006154
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0047201, 1.0025690, 1.0046303, -0.0008504, 0.0010021
1: -0.0005964, -0.0000879, -0.0006238, -0.0001102, -0.0002119, 0.0002497
2: -0.0095884, -0.0068935, -0.0094700, -0.0067480, -0.0013232, 0.0011229
3: 0.0018645, 0.0030911, 0.0017983, 0.0030372, -0.0005111, 0.0006023
4: -0.0013279, -0.0008063, -0.0013050, -0.0007782, -0.0002561, 0.0002173
5: -0.0131003, -0.0097108, -0.0129513, -0.0095277, -0.0016643, 0.0014123
6: 0.0040055, 0.0048658, 0.0039591, 0.0048280, -0.0003585, 0.0004224
7: 0.0072259, 0.0094518, 0.0071057, 0.0093539, -0.0009274, 0.0010929
8: 0.0042359, 0.0054064, 0.0041727, 0.0053550, -0.0004877, 0.0005748
9: -0.0081329, -0.0067756, -0.0080732, -0.0067023, -0.0006665, 0.0005655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0046310, 1.0025338, 1.0046316, -0.0008814, 0.0010462
1: -0.0006289, -0.0001100, -0.0006326, -0.0001098, -0.0002196, 0.0002607
2: -0.0094709, -0.0067212, -0.0094719, -0.0067015, -0.0013815, 0.0011639
3: 0.0017861, 0.0030376, 0.0017771, 0.0030381, -0.0005298, 0.0006288
4: -0.0013052, -0.0007730, -0.0013054, -0.0007692, -0.0002674, 0.0002253
5: -0.0129525, -0.0094941, -0.0129537, -0.0094693, -0.0017375, 0.0014639
6: 0.0039505, 0.0048283, 0.0039442, 0.0048286, -0.0003716, 0.0004410
7: 0.0070836, 0.0093547, 0.0070673, 0.0093555, -0.0009613, 0.0011410
8: 0.0041610, 0.0053554, 0.0041525, 0.0053558, -0.0005055, 0.0006001
9: -0.0080737, -0.0066888, -0.0080742, -0.0066789, -0.0006958, 0.0005862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005206, upper bound: 0.0006154
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.11 seconds
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005206, upper bound: 0.0006154
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
IS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005206, upper bound: 0.0006154
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0045161, -0.0010034, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001387, -0.0002500, 0.0002548
2: -0.0095382, -0.0068935, -0.0093191, -0.0069549, -0.0013503, 0.0013250
3: 0.0018645, 0.0030683, 0.0018925, 0.0029685, -0.0006031, 0.0006146
4: -0.0013182, -0.0008063, -0.0012758, -0.0008182, -0.0002613, 0.0002565
5: -0.0130371, -0.0097108, -0.0127615, -0.0097880, -0.0016983, 0.0016665
6: 0.0040055, 0.0048498, 0.0040251, 0.0047798, -0.0004230, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0092293, -0.0010944, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052894, -0.0005755, 0.0005865
9: -0.0081076, -0.0067756, -0.0079972, -0.0068065, -0.0006801, 0.0006673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006120
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0046059, 1.0026484, 1.0044328, -0.0009073, 0.0010989
1: -0.0006289, -0.0001163, -0.0006041, -0.0001594, -0.0002261, 0.0002738
2: -0.0094378, -0.0067212, -0.0092092, -0.0068528, -0.0014511, 0.0011981
3: 0.0017861, 0.0030226, 0.0018460, 0.0029185, -0.0005453, 0.0006605
4: -0.0012988, -0.0007730, -0.0012545, -0.0007985, -0.0002809, 0.0002319
5: -0.0129108, -0.0094941, -0.0126233, -0.0096596, -0.0018251, 0.0015069
6: 0.0039505, 0.0048177, 0.0039926, 0.0047448, -0.0003825, 0.0004632
7: 0.0070836, 0.0093273, 0.0071923, 0.0091385, -0.0009896, 0.0011985
8: 0.0041610, 0.0053410, 0.0042182, 0.0052417, -0.0005204, 0.0006303
9: -0.0080570, -0.0066888, -0.0079419, -0.0067551, -0.0007309, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0046310, 1.0026911, 1.0044833, -0.0009321, 0.0011101
1: -0.0006289, -0.0001100, -0.0005934, -0.0001468, -0.0002323, 0.0002766
2: -0.0094709, -0.0067212, -0.0092759, -0.0069093, -0.0014659, 0.0012308
3: 0.0017861, 0.0030376, 0.0018717, 0.0029489, -0.0005602, 0.0006672
4: -0.0013052, -0.0007730, -0.0012674, -0.0008094, -0.0002837, 0.0002382
5: -0.0129525, -0.0094941, -0.0127072, -0.0097306, -0.0018437, 0.0015480
6: 0.0039505, 0.0048283, 0.0040106, 0.0047661, -0.0003929, 0.0004680
7: 0.0070836, 0.0093547, 0.0072389, 0.0091936, -0.0010166, 0.0012107
8: 0.0041610, 0.0053554, 0.0042427, 0.0052707, -0.0005346, 0.0006367
9: -0.0080737, -0.0066888, -0.0079755, -0.0067835, -0.0007383, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006230
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0046278, -0.0008487, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001108, -0.0002115, 0.0002174
2: -0.0095382, -0.0068935, -0.0094667, -0.0067480, -0.0011524, 0.0011208
3: 0.0018645, 0.0030683, 0.0017983, 0.0030357, -0.0005101, 0.0005245
4: -0.0013182, -0.0008063, -0.0013044, -0.0007782, -0.0002230, 0.0002169
5: -0.0130371, -0.0097108, -0.0129471, -0.0095277, -0.0014494, 0.0014096
6: 0.0040055, 0.0048498, 0.0039591, 0.0048270, -0.0003578, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093512, -0.0009257, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053535, -0.0004868, 0.0005005
9: -0.0081076, -0.0067756, -0.0080715, -0.0067023, -0.0005804, 0.0005645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006120
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045973, 1.0025103, 1.0045332, -0.0007484, 0.0010138
1: -0.0006289, -0.0001184, -0.0006385, -0.0001344, -0.0001865, 0.0002526
2: -0.0094264, -0.0067212, -0.0093417, -0.0066705, -0.0013387, 0.0009882
3: 0.0017861, 0.0030174, 0.0017630, 0.0029788, -0.0004498, 0.0006093
4: -0.0012966, -0.0007730, -0.0012802, -0.0007632, -0.0002591, 0.0001913
5: -0.0128965, -0.0094941, -0.0127900, -0.0094302, -0.0016837, 0.0012429
6: 0.0039505, 0.0048141, 0.0039343, 0.0047871, -0.0003155, 0.0004274
7: 0.0070836, 0.0093179, 0.0070417, 0.0092480, -0.0008162, 0.0011057
8: 0.0041610, 0.0053361, 0.0041390, 0.0052993, -0.0004292, 0.0005815
9: -0.0080513, -0.0066888, -0.0080086, -0.0066632, -0.0006742, 0.0004977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004591, upper bound: 0.0003389
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002236, upper bound: 0.0003084
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0046310, 1.0025338, 1.0045983, -0.0007295, 0.0010462
1: -0.0006289, -0.0001100, -0.0006326, -0.0001182, -0.0001818, 0.0002607
2: -0.0094709, -0.0067212, -0.0094277, -0.0067015, -0.0013815, 0.0009634
3: 0.0017861, 0.0030376, 0.0017771, 0.0030180, -0.0004385, 0.0006288
4: -0.0013052, -0.0007730, -0.0012968, -0.0007692, -0.0002674, 0.0001865
5: -0.0129525, -0.0094941, -0.0128981, -0.0094693, -0.0017375, 0.0012116
6: 0.0039505, 0.0048283, 0.0039442, 0.0048145, -0.0003075, 0.0004410
7: 0.0070836, 0.0093547, 0.0070673, 0.0093190, -0.0007957, 0.0011410
8: 0.0041610, 0.0053554, 0.0041525, 0.0053366, -0.0004184, 0.0006001
9: -0.0080737, -0.0066888, -0.0080519, -0.0066789, -0.0006958, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006230
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.81 seconds
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006120
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006230
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006120
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0004591, upper bound: 0.0003389
IS_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0002236, upper bound: 0.0003084
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006230

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0025338, 1.0045983, -0.0007295, 0.0009133
1: -0.0006289, -0.0001182, -0.0006326, -0.0001182, -0.0001818, 0.0002276
2: -0.0094276, -0.0067212, -0.0094277, -0.0067015, -0.0012060, 0.0009634
3: 0.0017861, 0.0030179, 0.0017771, 0.0030180, -0.0004385, 0.0005489
4: -0.0012968, -0.0007730, -0.0012968, -0.0007692, -0.0002334, 0.0001865
5: -0.0128980, -0.0094941, -0.0128981, -0.0094693, -0.0015168, 0.0012116
6: 0.0039505, 0.0048145, 0.0039442, 0.0048145, -0.0003075, 0.0003850
7: 0.0070836, 0.0093189, 0.0070673, 0.0093190, -0.0007957, 0.0009960
8: 0.0041610, 0.0053366, 0.0041525, 0.0053366, -0.0004184, 0.0005238
9: -0.0080519, -0.0066888, -0.0080519, -0.0066789, -0.0006074, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0004744
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0005147
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.71 seconds
IS_A2_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
IS_A2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0004744
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0005147

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025103, 1.0045332, -0.0007178, 0.0009504
1: -0.0006342, -0.0001344, -0.0006385, -0.0001344, -0.0001789, 0.0002368
2: -0.0093416, -0.0066933, -0.0093417, -0.0066705, -0.0012550, 0.0009479
3: 0.0017734, 0.0029788, 0.0017630, 0.0029788, -0.0004314, 0.0005712
4: -0.0012802, -0.0007676, -0.0012802, -0.0007632, -0.0002429, 0.0001835
5: -0.0127898, -0.0094589, -0.0127900, -0.0094302, -0.0015785, 0.0011922
6: 0.0039416, 0.0047870, 0.0039343, 0.0047871, -0.0003026, 0.0004006
7: 0.0070605, 0.0092479, 0.0070417, 0.0092480, -0.0007829, 0.0010366
8: 0.0041489, 0.0052992, 0.0041390, 0.0052993, -0.0004117, 0.0005451
9: -0.0080086, -0.0066747, -0.0080086, -0.0066632, -0.0006321, 0.0004774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.79 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.87 seconds
IS_A2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
IS_A2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
IS_A2_B2_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0025338, 1.0045983, -0.0007295, 0.0009133
1: -0.0006289, -0.0001182, -0.0006326, -0.0001182, -0.0001818, 0.0002276
2: -0.0094276, -0.0067212, -0.0094277, -0.0067015, -0.0012060, 0.0009634
3: 0.0017861, 0.0030179, 0.0017771, 0.0030180, -0.0004385, 0.0005489
4: -0.0012968, -0.0007730, -0.0012968, -0.0007692, -0.0002334, 0.0001865
5: -0.0128980, -0.0094941, -0.0128981, -0.0094693, -0.0015168, 0.0012116
6: 0.0039505, 0.0048145, 0.0039442, 0.0048145, -0.0003075, 0.0003850
7: 0.0070836, 0.0093189, 0.0070673, 0.0093190, -0.0007957, 0.0009960
8: 0.0041610, 0.0053366, 0.0041525, 0.0053366, -0.0004184, 0.0005238
9: -0.0080519, -0.0066888, -0.0080519, -0.0066789, -0.0006074, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.80 seconds
IS_A2_B1_A1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
IS_A2_B2_A1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.80
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025103, 1.0045332, -0.0007178, 0.0009504
1: -0.0006342, -0.0001344, -0.0006385, -0.0001344, -0.0001789, 0.0002368
2: -0.0093416, -0.0066933, -0.0093417, -0.0066705, -0.0012550, 0.0009479
3: 0.0017734, 0.0029788, 0.0017630, 0.0029788, -0.0004314, 0.0005712
4: -0.0012802, -0.0007676, -0.0012802, -0.0007632, -0.0002429, 0.0001835
5: -0.0127898, -0.0094589, -0.0127900, -0.0094302, -0.0015785, 0.0011922
6: 0.0039416, 0.0047870, 0.0039343, 0.0047871, -0.0003026, 0.0004006
7: 0.0070605, 0.0092479, 0.0070417, 0.0092480, -0.0007829, 0.0010366
8: 0.0041489, 0.0052992, 0.0041390, 0.0052993, -0.0004117, 0.0005451
9: -0.0080086, -0.0066747, -0.0080086, -0.0066632, -0.0006321, 0.0004774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.81 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.00 seconds
IS_A2_B1_A1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
IS_A2_B2_A1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
IS_A2_B2_A2_B2_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0025338, 1.0045983, -0.0007295, 0.0009133
1: -0.0006289, -0.0001182, -0.0006326, -0.0001182, -0.0001818, 0.0002276
2: -0.0094276, -0.0067212, -0.0094277, -0.0067015, -0.0012060, 0.0009634
3: 0.0017861, 0.0030179, 0.0017771, 0.0030180, -0.0004385, 0.0005489
4: -0.0012968, -0.0007730, -0.0012968, -0.0007692, -0.0002334, 0.0001865
5: -0.0128980, -0.0094941, -0.0128981, -0.0094693, -0.0015168, 0.0012116
6: 0.0039505, 0.0048145, 0.0039442, 0.0048145, -0.0003075, 0.0003850
7: 0.0070836, 0.0093189, 0.0070673, 0.0093190, -0.0007957, 0.0009960
8: 0.0041610, 0.0053366, 0.0041525, 0.0053366, -0.0004184, 0.0005238
9: -0.0080519, -0.0066888, -0.0080519, -0.0066789, -0.0006074, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
time: 0.74 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.05 seconds
IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006004
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 3.05
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026484, 1.0044328, -0.0009054, 0.0010205
1: -0.0006342, -0.0001344, -0.0006041, -0.0001594, -0.0002256, 0.0002543
2: -0.0093416, -0.0066933, -0.0092092, -0.0068528, -0.0013475, 0.0011956
3: 0.0017734, 0.0029788, 0.0018460, 0.0029185, -0.0005442, 0.0006133
4: -0.0012802, -0.0007676, -0.0012545, -0.0007985, -0.0002608, 0.0002314
5: -0.0127898, -0.0094589, -0.0126233, -0.0096596, -0.0016948, 0.0015038
6: 0.0039416, 0.0047870, 0.0039926, 0.0047448, -0.0003817, 0.0004302
7: 0.0070605, 0.0092479, 0.0071923, 0.0091385, -0.0009875, 0.0011130
8: 0.0041489, 0.0052992, 0.0042182, 0.0052417, -0.0005193, 0.0005853
9: -0.0080086, -0.0066747, -0.0079419, -0.0067551, -0.0006787, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026484, 1.0044328, -0.0009073, 0.0011018
1: -0.0006289, -0.0001182, -0.0006041, -0.0001594, -0.0002261, 0.0002746
2: -0.0094276, -0.0067212, -0.0092092, -0.0068528, -0.0014550, 0.0011981
3: 0.0017861, 0.0030179, 0.0018460, 0.0029185, -0.0005453, 0.0006622
4: -0.0012968, -0.0007730, -0.0012545, -0.0007985, -0.0002816, 0.0002319
5: -0.0128980, -0.0094941, -0.0126233, -0.0096596, -0.0018300, 0.0015069
6: 0.0039505, 0.0048145, 0.0039926, 0.0047448, -0.0003825, 0.0004645
7: 0.0070836, 0.0093189, 0.0071923, 0.0091385, -0.0009896, 0.0012017
8: 0.0041610, 0.0053366, 0.0042182, 0.0052417, -0.0005204, 0.0006320
9: -0.0080519, -0.0066888, -0.0079419, -0.0067551, -0.0007328, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025103, 1.0045332, -0.0007178, 0.0009504
1: -0.0006342, -0.0001344, -0.0006385, -0.0001344, -0.0001789, 0.0002368
2: -0.0093416, -0.0066933, -0.0093417, -0.0066705, -0.0012550, 0.0009479
3: 0.0017734, 0.0029788, 0.0017630, 0.0029788, -0.0004314, 0.0005712
4: -0.0012802, -0.0007676, -0.0012802, -0.0007632, -0.0002429, 0.0001835
5: -0.0127898, -0.0094589, -0.0127900, -0.0094302, -0.0015785, 0.0011922
6: 0.0039416, 0.0047870, 0.0039343, 0.0047871, -0.0003026, 0.0004006
7: 0.0070605, 0.0092479, 0.0070417, 0.0092480, -0.0007829, 0.0010366
8: 0.0041489, 0.0052992, 0.0041390, 0.0052993, -0.0004117, 0.0005451
9: -0.0080086, -0.0066747, -0.0080086, -0.0066632, -0.0006321, 0.0004774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0025338, 1.0045983, -0.0008462, 0.0009443
1: -0.0006342, -0.0001344, -0.0006326, -0.0001182, -0.0002108, 0.0002353
2: -0.0093416, -0.0066933, -0.0094277, -0.0067015, -0.0012470, 0.0011174
3: 0.0017734, 0.0029788, 0.0017771, 0.0030180, -0.0005086, 0.0005676
4: -0.0012802, -0.0007676, -0.0012968, -0.0007692, -0.0002413, 0.0002163
5: -0.0127898, -0.0094589, -0.0128981, -0.0094693, -0.0015683, 0.0014054
6: 0.0039416, 0.0047870, 0.0039442, 0.0048145, -0.0003567, 0.0003981
7: 0.0070605, 0.0092479, 0.0070673, 0.0093190, -0.0009229, 0.0010299
8: 0.0041489, 0.0052992, 0.0041525, 0.0053366, -0.0004853, 0.0005416
9: -0.0080086, -0.0066747, -0.0080519, -0.0066789, -0.0006280, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
time: 0.83 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.01 seconds
IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002156, upper bound: 0.0004852
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001956, upper bound: 0.0002130
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002272, upper bound: 0.0005104
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0002251, upper bound: 0.0003264
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006004
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006229
IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
IS_A2_B2_A1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0004439, upper bound: 0.0002149
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0001940, upper bound: 0.0001940
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.01
Output dim: 0, lower bound: -0.0005236, upper bound: 0.0006003

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0026886, 1.0044327, -0.0009056, 0.0011034
1: -0.0005964, -0.0000973, -0.0005940, -0.0001595, -0.0002256, 0.0002749
2: -0.0095382, -0.0068935, -0.0092089, -0.0069061, -0.0014570, 0.0011958
3: 0.0018645, 0.0030683, 0.0018702, 0.0029184, -0.0005443, 0.0006632
4: -0.0013182, -0.0008063, -0.0012545, -0.0008088, -0.0002820, 0.0002314
5: -0.0130371, -0.0097108, -0.0126229, -0.0097266, -0.0018326, 0.0015040
6: 0.0040055, 0.0048498, 0.0040096, 0.0047447, -0.0003817, 0.0004651
7: 0.0072259, 0.0094103, 0.0072363, 0.0091383, -0.0009877, 0.0012034
8: 0.0042359, 0.0053846, 0.0042414, 0.0052416, -0.0005194, 0.0006329
9: -0.0081076, -0.0067756, -0.0079417, -0.0067819, -0.0007338, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0027256, 1.0044831, -0.0009083, 0.0010226
1: -0.0005964, -0.0000973, -0.0005848, -0.0001469, -0.0002263, 0.0002548
2: -0.0095382, -0.0068935, -0.0092756, -0.0069549, -0.0013503, 0.0011994
3: 0.0018645, 0.0030683, 0.0018925, 0.0029487, -0.0005459, 0.0006146
4: -0.0013182, -0.0008063, -0.0012674, -0.0008182, -0.0002613, 0.0002321
5: -0.0130371, -0.0097108, -0.0127068, -0.0097880, -0.0016983, 0.0015085
6: 0.0040055, 0.0048498, 0.0040251, 0.0047659, -0.0003829, 0.0004311
7: 0.0072259, 0.0094103, 0.0072766, 0.0091933, -0.0009906, 0.0011153
8: 0.0042359, 0.0053846, 0.0042626, 0.0052705, -0.0005209, 0.0005865
9: -0.0081076, -0.0067756, -0.0079753, -0.0068065, -0.0006801, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005891
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005064, upper bound: 0.0006120
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025275, 1.0045331, 1.0026911, 1.0044833, -0.0010257, 0.0010040
1: -0.0006342, -0.0001344, -0.0005934, -0.0001468, -0.0002556, 0.0002502
2: -0.0093416, -0.0066933, -0.0092759, -0.0069093, -0.0013258, 0.0013544
3: 0.0017734, 0.0029788, 0.0018717, 0.0029489, -0.0006165, 0.0006035
4: -0.0012802, -0.0007676, -0.0012674, -0.0008094, -0.0002566, 0.0002621
5: -0.0127898, -0.0094589, -0.0127072, -0.0097306, -0.0016676, 0.0017035
6: 0.0039416, 0.0047870, 0.0040106, 0.0047661, -0.0004324, 0.0004232
7: 0.0070605, 0.0092479, 0.0072389, 0.0091936, -0.0011187, 0.0010951
8: 0.0041489, 0.0052992, 0.0042427, 0.0052707, -0.0005883, 0.0005759
9: -0.0080086, -0.0066747, -0.0079755, -0.0067835, -0.0006678, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006003
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025487, 1.0045983, 1.0026911, 1.0044833, -0.0009321, 0.0010024
1: -0.0006289, -0.0001182, -0.0005934, -0.0001468, -0.0002323, 0.0002498
2: -0.0094276, -0.0067212, -0.0092759, -0.0069093, -0.0013236, 0.0012308
3: 0.0017861, 0.0030179, 0.0018717, 0.0029489, -0.0005602, 0.0006024
4: -0.0012968, -0.0007730, -0.0012674, -0.0008094, -0.0002562, 0.0002382
5: -0.0128980, -0.0094941, -0.0127072, -0.0097306, -0.0016647, 0.0015480
6: 0.0039505, 0.0048145, 0.0040106, 0.0047661, -0.0003929, 0.0004225
7: 0.0070836, 0.0093189, 0.0072389, 0.0091936, -0.0010166, 0.0010932
8: 0.0041610, 0.0053366, 0.0042427, 0.0052707, -0.0005346, 0.0005749
9: -0.0080519, -0.0066888, -0.0079755, -0.0067835, -0.0006666, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005205, upper bound: 0.0006154
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0006230
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002184, upper bound: 0.0000676
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025690, 1.0045980, -0.0007061, 0.0008727
1: -0.0005964, -0.0000973, -0.0006238, -0.0001182, -0.0001759, 0.0002174
2: -0.0095382, -0.0068935, -0.0094274, -0.0067480, -0.0011524, 0.0009324
3: 0.0018645, 0.0030683, 0.0017983, 0.0030178, -0.0004244, 0.0005245
4: -0.0013182, -0.0008063, -0.0012968, -0.0007782, -0.0002230, 0.0001805
5: -0.0130371, -0.0097108, -0.0128977, -0.0095277, -0.0014494, 0.0011727
6: 0.0040055, 0.0048498, 0.0039591, 0.0048144, -0.0002976, 0.0003679
7: 0.0072259, 0.0094103, 0.0071057, 0.0093187, -0.0007701, 0.0009518
8: 0.0042359, 0.0053846, 0.0041727, 0.0053365, -0.0004050, 0.0005005
9: -0.0081076, -0.0067756, -0.0080518, -0.0067023, -0.0005804, 0.0004696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005891
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0006032
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026791, 1.0046819, 1.0025522, 1.0045329, -0.0007453, 0.0009968
1: -0.0005964, -0.0000973, -0.0006280, -0.0001345, -0.0001857, 0.0002484
2: -0.0095382, -0.0068935, -0.0093414, -0.0067258, -0.0013162, 0.0009841
3: 0.0018645, 0.0030683, 0.0017882, 0.0029787, -0.0004479, 0.0005991
4: -0.0013182, -0.0008063, -0.0012801, -0.0007739, -0.0002548, 0.0001905
5: -0.0130371, -0.0097108, -0.0127896, -0.0094999, -0.0016555, 0.0012377
6: 0.0040055, 0.0048498, 0.0039520, 0.0047870, -0.0003142, 0.0004202
7: 0.0072259, 0.0094103, 0.0070874, 0.0092477, -0.0008128, 0.0010871
8: 0.0042359, 0.0053846, 0.0041631, 0.0052991, -0.0004274, 0.0005717
9: -0.0081076, -0.0067756, -0.0080085, -0.0066911, -0.0006629, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.90 + 598.27 = 601.17 seconds
