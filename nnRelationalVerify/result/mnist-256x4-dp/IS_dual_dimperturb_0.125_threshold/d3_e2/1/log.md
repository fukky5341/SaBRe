## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00088668


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0011732, 0.0021522, 0.0011732, 0.0021522, -0.0006275, 0.0006275)
1: (0.0014918, 0.0016332, 0.0014918, 0.0016332, -0.0000907, 0.0000907)
2: (0.0131700, 0.0137112, 0.0131700, 0.0137112, -0.0003469, 0.0003469)
3: (-0.0010595, -0.0004996, -0.0010595, -0.0004996, -0.0003588, 0.0003588)
4: (-0.0034961, -0.0028900, -0.0034961, -0.0028900, -0.0003884, 0.0003884)
5: (0.0068483, 0.0074218, 0.0068483, 0.0074218, -0.0003676, 0.0003676)
6: (0.0048717, 0.0071472, 0.0048717, 0.0071472, -0.0014585, 0.0014585)
7: (-0.0122906, -0.0091916, -0.0122906, -0.0091916, -0.0019864, 0.0019864)
8: (0.9805561, 0.9827392, 0.9805561, 0.9827392, -0.0013992, 0.0013992)
9: (-0.0002190, 0.0017626, -0.0002190, 0.0017626, -0.0012701, 0.0012701)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 1.27 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0010823, upper bound: 0.0010823

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010225
time: 0.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207
time: 0.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010225
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0011718, 0.0021158, 0.0011732, 0.0021472, -0.0005741, 0.0005697
1: 0.0014916, 0.0016280, 0.0014918, 0.0016325, -0.0000829, 0.0000823
2: 0.0131901, 0.0137120, 0.0131727, 0.0137112, -0.0003150, 0.0003174
3: -0.0010386, -0.0004989, -0.0010566, -0.0004997, -0.0003257, 0.0003283
4: -0.0034969, -0.0029126, -0.0034961, -0.0028931, -0.0003554, 0.0003526
5: 0.0068696, 0.0074226, 0.0068512, 0.0074218, -0.0003337, 0.0003363
6: 0.0049563, 0.0071503, 0.0048832, 0.0071472, -0.0013241, 0.0013343
7: -0.0122949, -0.0093068, -0.0122906, -0.0092072, -0.0018172, 0.0018033
8: 0.9805531, 0.9826580, 0.9805561, 0.9827281, -0.0012801, 0.0012703
9: -0.0001453, 0.0017653, -0.0002090, 0.0017626, -0.0011531, 0.0011620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009512, upper bound: 0.0009283
time: 0.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009480, upper bound: 0.0009449
time: 0.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0011733, 0.0021225, 0.0011732, 0.0021488, -0.0006247, 0.0005255
1: 0.0014918, 0.0016289, 0.0014918, 0.0016327, -0.0000902, 0.0000759
2: 0.0131864, 0.0137112, 0.0131719, 0.0137112, -0.0002905, 0.0003454
3: -0.0010425, -0.0004997, -0.0010575, -0.0004997, -0.0003005, 0.0003572
4: -0.0034960, -0.0029084, -0.0034961, -0.0028922, -0.0003867, 0.0003253
5: 0.0068657, 0.0074218, 0.0068503, 0.0074218, -0.0003078, 0.0003659
6: 0.0049407, 0.0071470, 0.0048797, 0.0071472, -0.0012215, 0.0014519
7: -0.0122903, -0.0092855, -0.0122906, -0.0092024, -0.0019773, 0.0016635
8: 0.9805563, 0.9826730, 0.9805561, 0.9827315, -0.0013929, 0.0011718
9: -0.0001589, 0.0017624, -0.0002121, 0.0017626, -0.0010637, 0.0012644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009512, upper bound: 0.0009316
time: 0.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009480, upper bound: 0.0009480
time: 0.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 8, lower bound: -0.0009512, upper bound: 0.0009283
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 8, lower bound: -0.0009480, upper bound: 0.0009449
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 8, lower bound: -0.0009512, upper bound: 0.0009316
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 8, lower bound: -0.0009480, upper bound: 0.0009480

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0011718, 0.0021158, 0.0011985, 0.0021472, -0.0005741, 0.0005440
1: 0.0014916, 0.0016280, 0.0014955, 0.0016325, -0.0000829, 0.0000786
2: 0.0131901, 0.0137120, 0.0131727, 0.0136972, -0.0003007, 0.0003174
3: -0.0010386, -0.0004989, -0.0010566, -0.0005141, -0.0003110, 0.0003283
4: -0.0034969, -0.0029126, -0.0034804, -0.0028931, -0.0003554, 0.0003367
5: 0.0068696, 0.0074226, 0.0068512, 0.0074070, -0.0003186, 0.0003363
6: 0.0049563, 0.0071503, 0.0048832, 0.0070883, -0.0012643, 0.0013343
7: -0.0122949, -0.0093068, -0.0122103, -0.0092072, -0.0018172, 0.0017219
8: 0.9805531, 0.9826580, 0.9806127, 0.9827281, -0.0012801, 0.0012129
9: -0.0001453, 0.0017653, -0.0002090, 0.0017113, -0.0011010, 0.0011620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009283
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009283
time: 0.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0011826, 0.0021158, 0.0012343, 0.0022590, -0.0007234, 0.0005633
1: 0.0014932, 0.0016280, 0.0015006, 0.0016487, -0.0001045, 0.0000814
2: 0.0131901, 0.0137060, 0.0131109, 0.0136775, -0.0003114, 0.0004000
3: -0.0010386, -0.0005050, -0.0011205, -0.0005346, -0.0003221, 0.0004137
4: -0.0034902, -0.0029126, -0.0034582, -0.0028239, -0.0004478, 0.0003487
5: 0.0068696, 0.0074163, 0.0067857, 0.0073860, -0.0003300, 0.0004238
6: 0.0049563, 0.0071253, 0.0046235, 0.0070052, -0.0013093, 0.0016815
7: -0.0122607, -0.0093068, -0.0120972, -0.0088535, -0.0022900, 0.0017831
8: 0.9805772, 0.9826580, 0.9806923, 0.9829773, -0.0016131, 0.0012561
9: -0.0001453, 0.0017435, -0.0004351, 0.0016389, -0.0011402, 0.0014643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009449
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009449
time: 0.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0011733, 0.0021225, 0.0011985, 0.0021488, -0.0006247, 0.0004950
1: 0.0014918, 0.0016289, 0.0014955, 0.0016327, -0.0000902, 0.0000715
2: 0.0131864, 0.0137112, 0.0131719, 0.0136972, -0.0002736, 0.0003454
3: -0.0010425, -0.0004997, -0.0010575, -0.0005141, -0.0002830, 0.0003572
4: -0.0034960, -0.0029084, -0.0034804, -0.0028922, -0.0003867, 0.0003064
5: 0.0068657, 0.0074218, 0.0068503, 0.0074070, -0.0002899, 0.0003659
6: 0.0049407, 0.0071470, 0.0048797, 0.0070883, -0.0011504, 0.0014519
7: -0.0122903, -0.0092855, -0.0122104, -0.0092024, -0.0019773, 0.0015668
8: 0.9805563, 0.9826730, 0.9806127, 0.9827315, -0.0013929, 0.0011037
9: -0.0001589, 0.0017624, -0.0002121, 0.0017113, -0.0010018, 0.0012644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009314
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009314
time: 0.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0011849, 0.0021225, 0.0012343, 0.0022610, -0.0007632, 0.0005149
1: 0.0014935, 0.0016289, 0.0015006, 0.0016490, -0.0001103, 0.0000744
2: 0.0131864, 0.0137047, 0.0131098, 0.0136775, -0.0002847, 0.0004219
3: -0.0010425, -0.0005064, -0.0011217, -0.0005346, -0.0002944, 0.0004364
4: -0.0034888, -0.0029084, -0.0034583, -0.0028227, -0.0004724, 0.0003187
5: 0.0068657, 0.0074149, 0.0067845, 0.0073860, -0.0003016, 0.0004471
6: 0.0049407, 0.0071199, 0.0046187, 0.0070052, -0.0011968, 0.0017738
7: -0.0122534, -0.0092855, -0.0120972, -0.0088470, -0.0024158, 0.0016300
8: 0.9805824, 0.9826729, 0.9806923, 0.9829818, -0.0017017, 0.0011482
9: -0.0001589, 0.0017388, -0.0004393, 0.0016390, -0.0010422, 0.0015447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009480
time: 0.48 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009480
time: 0.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009283
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009283
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009449
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009449
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009314
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009314, upper bound: 0.0009314
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009480
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 8, lower bound: -0.0009420, upper bound: 0.0009480

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0011985, 0.0021472, -0.0005439, 0.0005440
1: 0.0014955, 0.0016280, 0.0014955, 0.0016325, -0.0000786, 0.0000786
2: 0.0131901, 0.0136970, 0.0131727, 0.0136972, -0.0003007, 0.0003007
3: -0.0010386, -0.0005144, -0.0010566, -0.0005141, -0.0003110, 0.0003110
4: -0.0034801, -0.0029126, -0.0034804, -0.0028931, -0.0003367, 0.0003367
5: 0.0068696, 0.0074067, 0.0068512, 0.0074070, -0.0003186, 0.0003186
6: 0.0049563, 0.0070874, 0.0048832, 0.0070883, -0.0012643, 0.0012642
7: -0.0122091, -0.0093068, -0.0122103, -0.0092072, -0.0017217, 0.0017219
8: 0.9806135, 0.9826580, 0.9806127, 0.9827281, -0.0012128, 0.0012129
9: -0.0001453, 0.0017105, -0.0002090, 0.0017113, -0.0011010, 0.0011009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012325, 0.0022261, 0.0011985, 0.0021472, -0.0005587, 0.0007089
1: 0.0015004, 0.0016439, 0.0014955, 0.0016325, -0.0000807, 0.0001024
2: 0.0131291, 0.0136784, 0.0131727, 0.0136972, -0.0003919, 0.0003089
3: -0.0011017, -0.0005336, -0.0010566, -0.0005141, -0.0004054, 0.0003195
4: -0.0034593, -0.0028443, -0.0034804, -0.0028931, -0.0003459, 0.0004388
5: 0.0068050, 0.0073870, 0.0068512, 0.0074070, -0.0004153, 0.0003273
6: 0.0046999, 0.0070092, 0.0048832, 0.0070883, -0.0016477, 0.0012986
7: -0.0121027, -0.0089576, -0.0122103, -0.0092072, -0.0017686, 0.0022441
8: 0.9806885, 0.9829040, 0.9806127, 0.9827281, -0.0012459, 0.0015808
9: -0.0003687, 0.0016424, -0.0002090, 0.0017113, -0.0014349, 0.0011309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0011826, 0.0021158, 0.0012325, 0.0022261, -0.0006868, 0.0005145
1: 0.0014932, 0.0016280, 0.0015004, 0.0016439, -0.0000992, 0.0000743
2: 0.0131901, 0.0137060, 0.0131291, 0.0136784, -0.0002845, 0.0003797
3: -0.0010386, -0.0005050, -0.0011017, -0.0005336, -0.0002942, 0.0003927
4: -0.0034902, -0.0029126, -0.0034593, -0.0028443, -0.0004252, 0.0003185
5: 0.0068696, 0.0074163, 0.0068050, 0.0073870, -0.0003014, 0.0004023
6: 0.0049563, 0.0071253, 0.0046999, 0.0070092, -0.0011959, 0.0015964
7: -0.0122607, -0.0093068, -0.0121027, -0.0089576, -0.0021741, 0.0016288
8: 0.9805772, 0.9826580, 0.9806885, 0.9829040, -0.0015315, 0.0011473
9: -0.0001453, 0.0017435, -0.0003687, 0.0016424, -0.0010415, 0.0013902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009449
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009283
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011826, 0.0021158, 0.0012343, 0.0022317, -0.0007131, 0.0005613
1: 0.0014932, 0.0016280, 0.0015006, 0.0016447, -0.0001030, 0.0000811
2: 0.0131901, 0.0137060, 0.0131260, 0.0136774, -0.0003104, 0.0003942
3: -0.0010386, -0.0005050, -0.0011049, -0.0005346, -0.0003210, 0.0004077
4: -0.0034902, -0.0029126, -0.0034582, -0.0028408, -0.0004414, 0.0003475
5: 0.0068696, 0.0074163, 0.0068017, 0.0073860, -0.0003288, 0.0004177
6: 0.0049563, 0.0071253, 0.0046868, 0.0070051, -0.0013047, 0.0016574
7: -0.0122607, -0.0093068, -0.0120970, -0.0089397, -0.0022572, 0.0017769
8: 0.9805772, 0.9826580, 0.9806924, 0.9829165, -0.0015900, 0.0012517
9: -0.0001453, 0.0017435, -0.0003800, 0.0016388, -0.0011362, 0.0014433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009449
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009283
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0011986, 0.0021225, 0.0011985, 0.0021488, -0.0005989, 0.0004949
1: 0.0014955, 0.0016289, 0.0014955, 0.0016327, -0.0000865, 0.0000715
2: 0.0131864, 0.0136972, 0.0131719, 0.0136972, -0.0002736, 0.0003311
3: -0.0010425, -0.0005142, -0.0010575, -0.0005141, -0.0002830, 0.0003425
4: -0.0034803, -0.0029084, -0.0034804, -0.0028922, -0.0003707, 0.0003064
5: 0.0068657, 0.0074069, 0.0068503, 0.0074070, -0.0002899, 0.0003509
6: 0.0049407, 0.0070881, 0.0048797, 0.0070883, -0.0011504, 0.0013921
7: -0.0122101, -0.0092855, -0.0122104, -0.0092024, -0.0018959, 0.0015667
8: 0.9806128, 0.9826729, 0.9806127, 0.9827315, -0.0013355, 0.0011036
9: -0.0001589, 0.0017111, -0.0002121, 0.0017113, -0.0010018, 0.0012123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012343, 0.0022317, 0.0011985, 0.0021488, -0.0005878, 0.0006600
1: 0.0015006, 0.0016447, 0.0014955, 0.0016327, -0.0000849, 0.0000954
2: 0.0131260, 0.0136774, 0.0131719, 0.0136972, -0.0003649, 0.0003250
3: -0.0011049, -0.0005346, -0.0010575, -0.0005141, -0.0003774, 0.0003361
4: -0.0034582, -0.0028408, -0.0034804, -0.0028922, -0.0003639, 0.0004086
5: 0.0068017, 0.0073860, 0.0068503, 0.0074070, -0.0003866, 0.0003443
6: 0.0046868, 0.0070051, 0.0048797, 0.0070883, -0.0015341, 0.0013663
7: -0.0120970, -0.0089397, -0.0122104, -0.0092024, -0.0018607, 0.0020893
8: 0.9806924, 0.9829165, 0.9806127, 0.9827315, -0.0013107, 0.0014717
9: -0.0003800, 0.0016388, -0.0002121, 0.0017113, -0.0013359, 0.0011898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0011849, 0.0021225, 0.0012325, 0.0022261, -0.0007231, 0.0005662
1: 0.0014935, 0.0016289, 0.0015004, 0.0016439, -0.0001045, 0.0000818
2: 0.0131864, 0.0137047, 0.0131291, 0.0136784, -0.0003130, 0.0003998
3: -0.0010425, -0.0005064, -0.0011017, -0.0005336, -0.0003238, 0.0004135
4: -0.0034888, -0.0029084, -0.0034593, -0.0028443, -0.0004476, 0.0003505
5: 0.0068657, 0.0074149, 0.0068050, 0.0073870, -0.0003317, 0.0004236
6: 0.0049407, 0.0071199, 0.0046999, 0.0070092, -0.0013160, 0.0016807
7: -0.0122534, -0.0092855, -0.0121027, -0.0089576, -0.0022889, 0.0017923
8: 0.9805824, 0.9826729, 0.9806885, 0.9829040, -0.0016124, 0.0012626
9: -0.0001589, 0.0017388, -0.0003687, 0.0016424, -0.0011461, 0.0014636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009480
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009314
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011849, 0.0021225, 0.0012343, 0.0022317, -0.0006859, 0.0005137
1: 0.0014935, 0.0016289, 0.0015006, 0.0016447, -0.0000991, 0.0000742
2: 0.0131864, 0.0137047, 0.0131260, 0.0136774, -0.0002840, 0.0003792
3: -0.0010425, -0.0005064, -0.0011049, -0.0005346, -0.0002938, 0.0003922
4: -0.0034888, -0.0029084, -0.0034582, -0.0028408, -0.0004246, 0.0003180
5: 0.0068657, 0.0074149, 0.0068017, 0.0073860, -0.0003010, 0.0004018
6: 0.0049407, 0.0071199, 0.0046868, 0.0070051, -0.0011941, 0.0015943
7: -0.0122534, -0.0092855, -0.0120970, -0.0089397, -0.0021713, 0.0016262
8: 0.9805824, 0.9826729, 0.9806924, 0.9829165, -0.0015295, 0.0011456
9: -0.0001589, 0.0017388, -0.0003800, 0.0016388, -0.0010399, 0.0013884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009480
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009314
time: 0.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009283
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009449
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009283
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009449
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009283
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009385, upper bound: 0.0009316
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009480
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009314
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009480
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0009245, upper bound: 0.0009314

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0011989, 0.0021158, -0.0004954, 0.0004954
1: 0.0014955, 0.0016280, 0.0014955, 0.0016280, -0.0000716, 0.0000716
2: 0.0131901, 0.0136970, 0.0131901, 0.0136970, -0.0002739, 0.0002739
3: -0.0010386, -0.0005144, -0.0010386, -0.0005144, -0.0002833, 0.0002833
4: -0.0034801, -0.0029126, -0.0034801, -0.0029126, -0.0003066, 0.0003066
5: 0.0068696, 0.0074067, 0.0068696, 0.0074067, -0.0002902, 0.0002902
6: 0.0049563, 0.0070874, 0.0049563, 0.0070874, -0.0011514, 0.0011514
7: -0.0122091, -0.0093068, -0.0122091, -0.0093068, -0.0015681, 0.0015681
8: 0.9806135, 0.9826580, 0.9806135, 0.9826580, -0.0011046, 0.0011046
9: -0.0001453, 0.0017105, -0.0001453, 0.0017105, -0.0010027, 0.0010027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009100
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008949
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0011986, 0.0021225, -0.0005470, 0.0005420
1: 0.0014955, 0.0016280, 0.0014955, 0.0016289, -0.0000790, 0.0000783
2: 0.0131901, 0.0136970, 0.0131864, 0.0136972, -0.0002996, 0.0003024
3: -0.0010386, -0.0005144, -0.0010425, -0.0005142, -0.0003099, 0.0003128
4: -0.0034801, -0.0029126, -0.0034803, -0.0029084, -0.0003386, 0.0003355
5: 0.0068696, 0.0074067, 0.0068657, 0.0074069, -0.0003175, 0.0003205
6: 0.0049563, 0.0070874, 0.0049407, 0.0070881, -0.0012597, 0.0012715
7: -0.0122091, -0.0093068, -0.0122101, -0.0092855, -0.0017316, 0.0017156
8: 0.9806135, 0.9826580, 0.9806128, 0.9826729, -0.0012198, 0.0012085
9: -0.0001453, 0.0017105, -0.0001589, 0.0017111, -0.0010970, 0.0011073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009060, upper bound: 0.0008887
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008949
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0012325, 0.0022261, 0.0011989, 0.0021158, -0.0005102, 0.0006603
1: 0.0015004, 0.0016439, 0.0014955, 0.0016280, -0.0000737, 0.0000954
2: 0.0131291, 0.0136784, 0.0131901, 0.0136970, -0.0003651, 0.0002821
3: -0.0011017, -0.0005336, -0.0010386, -0.0005144, -0.0003776, 0.0002917
4: -0.0034593, -0.0028443, -0.0034801, -0.0029126, -0.0003158, 0.0004088
5: 0.0068050, 0.0073870, 0.0068696, 0.0074067, -0.0003868, 0.0002989
6: 0.0046999, 0.0070092, 0.0049563, 0.0070874, -0.0015348, 0.0011858
7: -0.0121027, -0.0089576, -0.0122091, -0.0093068, -0.0016150, 0.0020903
8: 0.9806885, 0.9829040, 0.9806135, 0.9826580, -0.0011376, 0.0014724
9: -0.0003687, 0.0016424, -0.0001453, 0.0017105, -0.0013366, 0.0010326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008058, upper bound: 0.0006776
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007138, upper bound: 0.0006532
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0012325, 0.0022261, 0.0011986, 0.0021225, -0.0005619, 0.0007069
1: 0.0015004, 0.0016439, 0.0014955, 0.0016289, -0.0000812, 0.0001021
2: 0.0131291, 0.0136784, 0.0131864, 0.0136972, -0.0003908, 0.0003106
3: -0.0011017, -0.0005336, -0.0010425, -0.0005142, -0.0004042, 0.0003213
4: -0.0034593, -0.0028443, -0.0034803, -0.0029084, -0.0003478, 0.0004376
5: 0.0068050, 0.0073870, 0.0068657, 0.0074069, -0.0004141, 0.0003291
6: 0.0046999, 0.0070092, 0.0049407, 0.0070881, -0.0016431, 0.0013059
7: -0.0121027, -0.0089576, -0.0122101, -0.0092855, -0.0017785, 0.0022378
8: 0.9806885, 0.9829040, 0.9806128, 0.9826729, -0.0012528, 0.0015763
9: -0.0003687, 0.0016424, -0.0001589, 0.0017111, -0.0014309, 0.0011372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008058, upper bound: 0.0006776
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007138, upper bound: 0.0006532
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0012325, 0.0022261, -0.0006603, 0.0005102
1: 0.0014955, 0.0016280, 0.0015004, 0.0016439, -0.0000954, 0.0000737
2: 0.0131901, 0.0136970, 0.0131291, 0.0136784, -0.0002821, 0.0003651
3: -0.0010386, -0.0005144, -0.0011017, -0.0005336, -0.0002917, 0.0003776
4: -0.0034801, -0.0029126, -0.0034593, -0.0028443, -0.0004088, 0.0003158
5: 0.0068696, 0.0074067, 0.0068050, 0.0073870, -0.0002989, 0.0003868
6: 0.0049563, 0.0070874, 0.0046999, 0.0070092, -0.0011858, 0.0015348
7: -0.0122091, -0.0093068, -0.0121027, -0.0089576, -0.0020903, 0.0016150
8: 0.9806135, 0.9826580, 0.9806885, 0.9829040, -0.0014724, 0.0011376
9: -0.0001453, 0.0017105, -0.0003687, 0.0016424, -0.0010326, 0.0013366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007508
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005846, upper bound: 0.0005846
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012325, 0.0022261, 0.0012325, 0.0022261, -0.0005158, 0.0005158
1: 0.0015004, 0.0016439, 0.0015004, 0.0016439, -0.0000745, 0.0000745
2: 0.0131291, 0.0136784, 0.0131291, 0.0136784, -0.0002852, 0.0002852
3: -0.0011017, -0.0005336, -0.0011017, -0.0005336, -0.0002950, 0.0002950
4: -0.0034593, -0.0028443, -0.0034593, -0.0028443, -0.0003193, 0.0003193
5: 0.0068050, 0.0073870, 0.0068050, 0.0073870, -0.0003022, 0.0003022
6: 0.0046999, 0.0070092, 0.0046999, 0.0070092, -0.0011989, 0.0011989
7: -0.0121027, -0.0089576, -0.0121027, -0.0089576, -0.0016328, 0.0016328
8: 0.9806885, 0.9829040, 0.9806885, 0.9829040, -0.0011502, 0.0011502
9: -0.0003687, 0.0016424, -0.0003687, 0.0016424, -0.0010441, 0.0010441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007508
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005846, upper bound: 0.0005846
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0012343, 0.0022317, -0.0006866, 0.0005308
1: 0.0014955, 0.0016280, 0.0015006, 0.0016447, -0.0000992, 0.0000767
2: 0.0131901, 0.0136970, 0.0131260, 0.0136774, -0.0002935, 0.0003796
3: -0.0010386, -0.0005144, -0.0011049, -0.0005346, -0.0003035, 0.0003926
4: -0.0034801, -0.0029126, -0.0034582, -0.0028408, -0.0004250, 0.0003286
5: 0.0068696, 0.0074067, 0.0068017, 0.0073860, -0.0003110, 0.0004022
6: 0.0049563, 0.0070874, 0.0046868, 0.0070051, -0.0012338, 0.0015958
7: -0.0122091, -0.0093068, -0.0120970, -0.0089397, -0.0021734, 0.0016804
8: 0.9806135, 0.9826580, 0.9806924, 0.9829165, -0.0015310, 0.0011837
9: -0.0001453, 0.0017105, -0.0003800, 0.0016388, -0.0010745, 0.0013897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006641, upper bound: 0.0007508
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006057, upper bound: 0.0005848
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012325, 0.0022261, 0.0012343, 0.0022317, -0.0005675, 0.0005626
1: 0.0015004, 0.0016439, 0.0015006, 0.0016447, -0.0000820, 0.0000813
2: 0.0131291, 0.0136784, 0.0131260, 0.0136774, -0.0003111, 0.0003138
3: -0.0011017, -0.0005336, -0.0011049, -0.0005346, -0.0003217, 0.0003245
4: -0.0034593, -0.0028443, -0.0034582, -0.0028408, -0.0003513, 0.0003483
5: 0.0068050, 0.0073870, 0.0068017, 0.0073860, -0.0003296, 0.0003324
6: 0.0046999, 0.0070092, 0.0046868, 0.0070051, -0.0013077, 0.0013190
7: -0.0121027, -0.0089576, -0.0120970, -0.0089397, -0.0017964, 0.0017810
8: 0.9806885, 0.9829040, 0.9806924, 0.9829165, -0.0012654, 0.0012546
9: -0.0003687, 0.0016424, -0.0003800, 0.0016388, -0.0011388, 0.0011487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007556, upper bound: 0.0006559
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006057, upper bound: 0.0005848
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0011986, 0.0021225, 0.0011989, 0.0021158, -0.0005420, 0.0005470
1: 0.0014955, 0.0016289, 0.0014955, 0.0016280, -0.0000783, 0.0000790
2: 0.0131864, 0.0136972, 0.0131901, 0.0136970, -0.0003024, 0.0002996
3: -0.0010425, -0.0005142, -0.0010386, -0.0005144, -0.0003128, 0.0003099
4: -0.0034803, -0.0029084, -0.0034801, -0.0029126, -0.0003355, 0.0003386
5: 0.0068657, 0.0074069, 0.0068696, 0.0074067, -0.0003205, 0.0003175
6: 0.0049407, 0.0070881, 0.0049563, 0.0070874, -0.0012715, 0.0012597
7: -0.0122101, -0.0092855, -0.0122091, -0.0093068, -0.0017156, 0.0017316
8: 0.9806128, 0.9826729, 0.9806135, 0.9826580, -0.0012085, 0.0012198
9: -0.0001589, 0.0017111, -0.0001453, 0.0017105, -0.0011073, 0.0010970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009116
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008924
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0011986, 0.0021225, 0.0011986, 0.0021225, -0.0004938, 0.0004938
1: 0.0014955, 0.0016289, 0.0014955, 0.0016289, -0.0000713, 0.0000713
2: 0.0131864, 0.0136972, 0.0131864, 0.0136972, -0.0002730, 0.0002730
3: -0.0010425, -0.0005142, -0.0010425, -0.0005142, -0.0002823, 0.0002823
4: -0.0034803, -0.0029084, -0.0034803, -0.0029084, -0.0003056, 0.0003056
5: 0.0068657, 0.0074069, 0.0068657, 0.0074069, -0.0002892, 0.0002892
6: 0.0049407, 0.0070881, 0.0049407, 0.0070881, -0.0011476, 0.0011476
7: -0.0122101, -0.0092855, -0.0122101, -0.0092855, -0.0015630, 0.0015630
8: 0.9806128, 0.9826729, 0.9806128, 0.9826729, -0.0011010, 0.0011010
9: -0.0001589, 0.0017111, -0.0001589, 0.0017111, -0.0009994, 0.0009994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009116
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008924
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0012343, 0.0022317, 0.0011989, 0.0021158, -0.0005308, 0.0006866
1: 0.0015006, 0.0016447, 0.0014955, 0.0016280, -0.0000767, 0.0000992
2: 0.0131260, 0.0136774, 0.0131901, 0.0136970, -0.0003796, 0.0002935
3: -0.0011049, -0.0005346, -0.0010386, -0.0005144, -0.0003926, 0.0003035
4: -0.0034582, -0.0028408, -0.0034801, -0.0029126, -0.0003286, 0.0004250
5: 0.0068017, 0.0073860, 0.0068696, 0.0074067, -0.0004022, 0.0003110
6: 0.0046868, 0.0070051, 0.0049563, 0.0070874, -0.0015958, 0.0012338
7: -0.0120970, -0.0089397, -0.0122091, -0.0093068, -0.0016804, 0.0021734
8: 0.9806924, 0.9829165, 0.9806135, 0.9826580, -0.0011837, 0.0015310
9: -0.0003800, 0.0016388, -0.0001453, 0.0017105, -0.0013897, 0.0010745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008063, upper bound: 0.0007065
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007139, upper bound: 0.0006938
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0012343, 0.0022317, 0.0011986, 0.0021225, -0.0005088, 0.0006588
1: 0.0015006, 0.0016447, 0.0014955, 0.0016289, -0.0000735, 0.0000952
2: 0.0131260, 0.0136774, 0.0131864, 0.0136972, -0.0003643, 0.0002813
3: -0.0011049, -0.0005346, -0.0010425, -0.0005142, -0.0003767, 0.0002909
4: -0.0034582, -0.0028408, -0.0034803, -0.0029084, -0.0003149, 0.0004078
5: 0.0068017, 0.0073860, 0.0068657, 0.0074069, -0.0003859, 0.0002980
6: 0.0046868, 0.0070051, 0.0049407, 0.0070881, -0.0015313, 0.0011825
7: -0.0120970, -0.0089397, -0.0122101, -0.0092855, -0.0016105, 0.0020855
8: 0.9806924, 0.9829165, 0.9806128, 0.9826729, -0.0011345, 0.0014691
9: -0.0003800, 0.0016388, -0.0001589, 0.0017111, -0.0013335, 0.0010298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008063, upper bound: 0.0007065
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007139, upper bound: 0.0006938
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0011986, 0.0021225, 0.0012325, 0.0022261, -0.0007069, 0.0005619
1: 0.0014955, 0.0016289, 0.0015004, 0.0016439, -0.0001021, 0.0000812
2: 0.0131864, 0.0136972, 0.0131291, 0.0136784, -0.0003106, 0.0003908
3: -0.0010425, -0.0005142, -0.0011017, -0.0005336, -0.0003213, 0.0004042
4: -0.0034803, -0.0029084, -0.0034593, -0.0028443, -0.0004376, 0.0003478
5: 0.0068657, 0.0074069, 0.0068050, 0.0073870, -0.0003291, 0.0004141
6: 0.0049407, 0.0070881, 0.0046999, 0.0070092, -0.0013059, 0.0016431
7: -0.0122101, -0.0092855, -0.0121027, -0.0089576, -0.0022378, 0.0017785
8: 0.9806128, 0.9826729, 0.9806885, 0.9829040, -0.0015763, 0.0012528
9: -0.0001589, 0.0017111, -0.0003687, 0.0016424, -0.0011372, 0.0014309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007556
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005848, upper bound: 0.0006057
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012343, 0.0022317, 0.0012325, 0.0022261, -0.0005626, 0.0005675
1: 0.0015006, 0.0016447, 0.0015004, 0.0016439, -0.0000813, 0.0000820
2: 0.0131260, 0.0136774, 0.0131291, 0.0136784, -0.0003138, 0.0003111
3: -0.0011049, -0.0005346, -0.0011017, -0.0005336, -0.0003245, 0.0003217
4: -0.0034582, -0.0028408, -0.0034593, -0.0028443, -0.0003483, 0.0003513
5: 0.0068017, 0.0073860, 0.0068050, 0.0073870, -0.0003324, 0.0003296
6: 0.0046868, 0.0070051, 0.0046999, 0.0070092, -0.0013190, 0.0013077
7: -0.0120970, -0.0089397, -0.0121027, -0.0089576, -0.0017810, 0.0017964
8: 0.9806924, 0.9829165, 0.9806885, 0.9829040, -0.0012546, 0.0012654
9: -0.0003800, 0.0016388, -0.0003687, 0.0016424, -0.0011487, 0.0011388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007556
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005848, upper bound: 0.0006057
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0011986, 0.0021225, 0.0012343, 0.0022317, -0.0006588, 0.0005088
1: 0.0014955, 0.0016289, 0.0015006, 0.0016447, -0.0000952, 0.0000735
2: 0.0131864, 0.0136972, 0.0131260, 0.0136774, -0.0002813, 0.0003643
3: -0.0010425, -0.0005142, -0.0011049, -0.0005346, -0.0002909, 0.0003767
4: -0.0034803, -0.0029084, -0.0034582, -0.0028408, -0.0004078, 0.0003149
5: 0.0068657, 0.0074069, 0.0068017, 0.0073860, -0.0002980, 0.0003859
6: 0.0049407, 0.0070881, 0.0046868, 0.0070051, -0.0011825, 0.0015313
7: -0.0122101, -0.0092855, -0.0120970, -0.0089397, -0.0020855, 0.0016105
8: 0.9806128, 0.9826729, 0.9806924, 0.9829165, -0.0014691, 0.0011345
9: -0.0001589, 0.0017111, -0.0003800, 0.0016388, -0.0010298, 0.0013335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006562, upper bound: 0.0007556
time: 0.51 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006006, upper bound: 0.0006153
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012343, 0.0022317, 0.0012343, 0.0022317, -0.0005150, 0.0005150
1: 0.0015006, 0.0016447, 0.0015006, 0.0016447, -0.0000744, 0.0000744
2: 0.0131260, 0.0136774, 0.0131260, 0.0136774, -0.0002848, 0.0002848
3: -0.0011049, -0.0005346, -0.0011049, -0.0005346, -0.0002945, 0.0002945
4: -0.0034582, -0.0028408, -0.0034582, -0.0028408, -0.0003188, 0.0003188
5: 0.0068017, 0.0073860, 0.0068017, 0.0073860, -0.0003017, 0.0003017
6: 0.0046868, 0.0070051, 0.0046868, 0.0070051, -0.0011971, 0.0011971
7: -0.0120970, -0.0089397, -0.0120970, -0.0089397, -0.0016303, 0.0016303
8: 0.9806924, 0.9829165, 0.9806924, 0.9829165, -0.0011484, 0.0011484
9: -0.0003800, 0.0016388, -0.0003800, 0.0016388, -0.0010425, 0.0010425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006562, upper bound: 0.0007556
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006006, upper bound: 0.0006153
time: 0.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009100
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008949
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0009060, upper bound: 0.0008887
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008949
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008058, upper bound: 0.0006776
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0007138, upper bound: 0.0006532
IS_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008058, upper bound: 0.0006776
IS_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0007138, upper bound: 0.0006532
IS_A1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007508
IS_A1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0005846, upper bound: 0.0005846
IS_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007508
IS_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0005846, upper bound: 0.0005846
IS_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006641, upper bound: 0.0007508
IS_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006057, upper bound: 0.0005848
IS_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0007556, upper bound: 0.0006559
IS_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006057, upper bound: 0.0005848
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009116
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008924
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0009116
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008919, upper bound: 0.0008924
IS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008063, upper bound: 0.0007065
IS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0007139, upper bound: 0.0006938
IS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0008063, upper bound: 0.0007065
IS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0007139, upper bound: 0.0006938
IS_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007556
IS_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0005848, upper bound: 0.0006057
IS_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006559, upper bound: 0.0007556
IS_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0005848, upper bound: 0.0006057
IS_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006562, upper bound: 0.0007556
IS_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006006, upper bound: 0.0006153
IS_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006562, upper bound: 0.0007556
IS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 8, lower bound: -0.0006006, upper bound: 0.0006153

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0012130, 0.0021158, 0.0011989, 0.0021158, -0.0004809, 0.0004954
1: 0.0014975, 0.0016280, 0.0014955, 0.0016280, -0.0000695, 0.0000716
2: 0.0131901, 0.0136892, 0.0131901, 0.0136970, -0.0002739, 0.0002659
3: -0.0010386, -0.0005224, -0.0010386, -0.0005144, -0.0002833, 0.0002750
4: -0.0034714, -0.0029126, -0.0034801, -0.0029126, -0.0002977, 0.0003066
5: 0.0068696, 0.0073985, 0.0068696, 0.0074067, -0.0002902, 0.0002817
6: 0.0049563, 0.0070547, 0.0049563, 0.0070874, -0.0011514, 0.0011176
7: -0.0121646, -0.0093068, -0.0122091, -0.0093068, -0.0015221, 0.0015681
8: 0.9806449, 0.9826580, 0.9806135, 0.9826580, -0.0010722, 0.0011046
9: -0.0001453, 0.0016820, -0.0001453, 0.0017105, -0.0010027, 0.0009733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008247, upper bound: 0.0007840
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007698, upper bound: 0.0007850
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012396, 0.0021722, 0.0012052, 0.0021158, -0.0004926, 0.0005738
1: 0.0015014, 0.0016361, 0.0014964, 0.0016280, -0.0000712, 0.0000829
2: 0.0131589, 0.0136745, 0.0131901, 0.0136935, -0.0003172, 0.0002723
3: -0.0010709, -0.0005377, -0.0010386, -0.0005180, -0.0003281, 0.0002817
4: -0.0034549, -0.0028776, -0.0034762, -0.0029126, -0.0003049, 0.0003552
5: 0.0068366, 0.0073829, 0.0068696, 0.0074030, -0.0003361, 0.0002886
6: 0.0048251, 0.0069927, 0.0049563, 0.0070727, -0.0013337, 0.0011449
7: -0.0120802, -0.0091281, -0.0121891, -0.0093068, -0.0015593, 0.0018163
8: 0.9807044, 0.9827839, 0.9806276, 0.9826580, -0.0010984, 0.0012795
9: -0.0002596, 0.0016280, -0.0001453, 0.0016977, -0.0011614, 0.0009971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008156, upper bound: 0.0007245
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007241, upper bound: 0.0007241
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0011989, 0.0021158, 0.0012140, 0.0021225, -0.0005470, 0.0005269
1: 0.0014955, 0.0016280, 0.0014977, 0.0016289, -0.0000790, 0.0000761
2: 0.0131901, 0.0136970, 0.0131864, 0.0136887, -0.0002913, 0.0003024
3: -0.0010386, -0.0005144, -0.0010425, -0.0005230, -0.0003013, 0.0003128
4: -0.0034801, -0.0029126, -0.0034708, -0.0029084, -0.0003386, 0.0003262
5: 0.0068696, 0.0074067, 0.0068657, 0.0073979, -0.0003087, 0.0003205
6: 0.0049563, 0.0070874, 0.0049407, 0.0070524, -0.0012248, 0.0012715
7: -0.0122091, -0.0093068, -0.0121615, -0.0092855, -0.0017316, 0.0016680
8: 0.9806135, 0.9826580, 0.9806470, 0.9826730, -0.0012198, 0.0011750
9: -0.0001453, 0.0017105, -0.0001589, 0.0016801, -0.0010666, 0.0011072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007773, upper bound: 0.0008193
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007787, upper bound: 0.0007629
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0012052, 0.0021158, 0.0012327, 0.0021862, -0.0006085, 0.0005390
1: 0.0014964, 0.0016280, 0.0015004, 0.0016382, -0.0000879, 0.0000779
2: 0.0131901, 0.0136935, 0.0131511, 0.0136784, -0.0002980, 0.0003364
3: -0.0010386, -0.0005180, -0.0010789, -0.0005337, -0.0003082, 0.0003479
4: -0.0034762, -0.0029126, -0.0034592, -0.0028690, -0.0003767, 0.0003337
5: 0.0068696, 0.0074030, 0.0068283, 0.0073870, -0.0003157, 0.0003564
6: 0.0049563, 0.0070727, 0.0047926, 0.0070090, -0.0012528, 0.0014143
7: -0.0121891, -0.0093068, -0.0121023, -0.0090838, -0.0019261, 0.0017062
8: 0.9806276, 0.9826580, 0.9806888, 0.9828151, -0.0013568, 0.0012019
9: -0.0001453, 0.0016977, -0.0002879, 0.0016422, -0.0010910, 0.0012316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007051, upper bound: 0.0008070
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007032, upper bound: 0.0007144
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0012140, 0.0021225, 0.0011989, 0.0021158, -0.0005269, 0.0005470
1: 0.0014977, 0.0016289, 0.0014955, 0.0016280, -0.0000761, 0.0000790
2: 0.0131864, 0.0136887, 0.0131901, 0.0136970, -0.0003024, 0.0002913
3: -0.0010425, -0.0005230, -0.0010386, -0.0005144, -0.0003128, 0.0003013
4: -0.0034708, -0.0029084, -0.0034801, -0.0029126, -0.0003262, 0.0003386
5: 0.0068657, 0.0073979, 0.0068696, 0.0074067, -0.0003205, 0.0003087
6: 0.0049407, 0.0070524, 0.0049563, 0.0070874, -0.0012715, 0.0012248
7: -0.0121615, -0.0092855, -0.0122091, -0.0093068, -0.0016680, 0.0017316
8: 0.9806470, 0.9826730, 0.9806135, 0.9826580, -0.0011750, 0.0012198
9: -0.0001589, 0.0016801, -0.0001453, 0.0017105, -0.0011072, 0.0010666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008193, upper bound: 0.0007773
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007629, upper bound: 0.0007787
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012327, 0.0021862, 0.0012052, 0.0021158, -0.0005390, 0.0006085
1: 0.0015004, 0.0016382, 0.0014964, 0.0016280, -0.0000779, 0.0000879
2: 0.0131511, 0.0136784, 0.0131901, 0.0136935, -0.0003364, 0.0002980
3: -0.0010789, -0.0005337, -0.0010386, -0.0005180, -0.0003479, 0.0003082
4: -0.0034592, -0.0028690, -0.0034762, -0.0029126, -0.0003337, 0.0003767
5: 0.0068283, 0.0073870, 0.0068696, 0.0074030, -0.0003564, 0.0003157
6: 0.0047926, 0.0070090, 0.0049563, 0.0070727, -0.0014143, 0.0012528
7: -0.0121023, -0.0090838, -0.0121891, -0.0093068, -0.0017062, 0.0019261
8: 0.9806888, 0.9828151, 0.9806276, 0.9826580, -0.0012019, 0.0013568
9: -0.0002879, 0.0016422, -0.0001453, 0.0016977, -0.0012316, 0.0010910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008070, upper bound: 0.0007051
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007144, upper bound: 0.0007032
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0012140, 0.0021225, 0.0011986, 0.0021225, -0.0004795, 0.0004938
1: 0.0014977, 0.0016289, 0.0014955, 0.0016289, -0.0000693, 0.0000713
2: 0.0131864, 0.0136887, 0.0131864, 0.0136972, -0.0002730, 0.0002651
3: -0.0010425, -0.0005230, -0.0010425, -0.0005142, -0.0002823, 0.0002742
4: -0.0034708, -0.0029084, -0.0034803, -0.0029084, -0.0002968, 0.0003056
5: 0.0068657, 0.0073979, 0.0068657, 0.0074069, -0.0002892, 0.0002809
6: 0.0049407, 0.0070524, 0.0049407, 0.0070881, -0.0011476, 0.0011146
7: -0.0121615, -0.0092855, -0.0122101, -0.0092855, -0.0015179, 0.0015630
8: 0.9806470, 0.9826730, 0.9806128, 0.9826729, -0.0010693, 0.0011010
9: -0.0001589, 0.0016801, -0.0001589, 0.0017111, -0.0009994, 0.0009706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008079, upper bound: 0.0007773
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007565, upper bound: 0.0007787
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012327, 0.0021862, 0.0012042, 0.0021225, -0.0004982, 0.0005724
1: 0.0015004, 0.0016382, 0.0014963, 0.0016289, -0.0000720, 0.0000827
2: 0.0131511, 0.0136784, 0.0131864, 0.0136941, -0.0003164, 0.0002754
3: -0.0010789, -0.0005337, -0.0010425, -0.0005174, -0.0003273, 0.0002848
4: -0.0034592, -0.0028690, -0.0034768, -0.0029084, -0.0003084, 0.0003543
5: 0.0068283, 0.0073870, 0.0068657, 0.0074036, -0.0003353, 0.0002918
6: 0.0047926, 0.0070090, 0.0049407, 0.0070751, -0.0013303, 0.0011578
7: -0.0121023, -0.0090838, -0.0121923, -0.0092855, -0.0015769, 0.0018118
8: 0.9806888, 0.9828151, 0.9806253, 0.9826729, -0.0011108, 0.0012762
9: -0.0002879, 0.0016422, -0.0001589, 0.0016998, -0.0011585, 0.0010083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007972, upper bound: 0.0007051
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007032, upper bound: 0.0007032
time: 0.46 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0008247, upper bound: 0.0007840
IS_A1_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007698, upper bound: 0.0007850
IS_A1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0008156, upper bound: 0.0007245
IS_A1_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007241, upper bound: 0.0007241
IS_A1_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007773, upper bound: 0.0008193
IS_A1_B1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007787, upper bound: 0.0007629
IS_A1_B1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007051, upper bound: 0.0008070
IS_A1_B1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007032, upper bound: 0.0007144
IS_A2_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0008193, upper bound: 0.0007773
IS_A2_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007629, upper bound: 0.0007787
IS_A2_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0008070, upper bound: 0.0007051
IS_A2_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007144, upper bound: 0.0007032
IS_A2_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0008079, upper bound: 0.0007773
IS_A2_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007565, upper bound: 0.0007787
IS_A2_B1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007972, upper bound: 0.0007051
IS_A2_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0007032, upper bound: 0.0007032

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.89 + 99.84 = 102.73 seconds
